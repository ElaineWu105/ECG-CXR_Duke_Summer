#!/usr/bin/env python3
"""Primary InfoNCE + auxiliary same-study multiview + disease prototypes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "7.15"))
sys.path.insert(2, str(EXP))

import engine  # noqa: E402
import run_experiments as runner  # noqa: E402
from losses import batch_retrieval_metrics, cross_patient_loss  # noqa: E402
from train_latest_gated_history import LatestGatedHistoryModel  # noqa: E402
from train_latest_gated_history_multiview import (  # noqa: E402
    MultiViewDataset, multiview_collate,
)

LABEL_COLUMNS = (
    "Pneumonia", "Consolidation", "Lung Opacity",
    "Pneumothorax", "Pleural Effusion", "Edema",
)
LAMBDA_MULTIVIEW = float(os.environ.get("LAMBDA_MULTIVIEW", "0.1"))
LAMBDA_PROTOTYPE = float(os.environ.get("LAMBDA_PROTOTYPE", "0.1"))


class PrototypeDataset(MultiViewDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = int(item["c2_row"])
        item["disease_targets"] = torch.from_numpy(self.data.labels_by_primary[row][0])
        item["disease_observed"] = torch.from_numpy(self.data.labels_by_primary[row][1])
        return item


def prototype_collate(batch):
    output = multiview_collate(batch)
    output["disease_targets"] = torch.stack([x["disease_targets"] for x in batch])
    output["disease_observed"] = torch.stack([x["disease_observed"] for x in batch])
    return output


def primary_loss(logits, patient_ids):
    loss, _ = cross_patient_loss(logits, patient_ids)
    return loss


def multiview_auxiliary_loss(model, query, positive_feats, positive_mask, patient_ids):
    """Same-study multi-positive loss, evaluated only for rows with >1 view."""
    batch_size, n_views, feature_dim = positive_feats.shape
    projected = model.cxr_proj(positive_feats.reshape(-1, feature_dim)).reshape(
        batch_size, n_views, -1)
    candidates = projected.reshape(batch_size * n_views, -1)
    candidate_valid = positive_mask.reshape(-1)
    owners = torch.arange(batch_size, device=query.device).repeat_interleave(n_views)
    candidate_patients = patient_ids[owners]
    logits = (query @ candidates.t()) * torch.exp(model.logit_scale).clamp(max=100.0)
    own_study = owners[None, :].eq(torch.arange(batch_size, device=query.device)[:, None])
    same_patient = patient_ids[:, None].eq(candidate_patients[None, :])
    valid = candidate_valid[None, :] & ((~same_patient) | own_study)
    positive = candidate_valid[None, :] & own_study
    log_den = torch.logsumexp(logits.masked_fill(~valid, float("-inf")), dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    use = positive_mask.sum(1) > 1
    if not use.any():
        return logits.new_zeros(()), 0
    return (-(log_num - log_den))[use].mean(), int(use.sum())


def prototype_auxiliary_loss(model, query, prototype_raw, targets, observed):
    prototypes = model.cxr_proj(prototype_raw)
    similarities = query @ prototypes.t()
    positive = observed.bool() & targets.bool()
    use = positive.any(1)
    if not use.any():
        return similarities.new_zeros(()), 0
    mean_positive_similarity = ((similarities * positive.float()).sum(1)
                                / positive.sum(1).clamp_min(1))
    return (1.0 - mean_positive_similarity)[use].mean(), int(use.sum())


def run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm,
              temporal_min_horizon_hours=None, temporal_max_horizon_hours=None,
              epoch=0, global_step_start=0, iteration_records=None, dynamics_log_every=1):
    del w_cross, w_temporal, temporal_min_horizon_hours, temporal_max_horizon_hours
    model.train()
    prototype_raw = loader.dataset.data.disease_prototypes.to(device)
    sums = {"loss": 0.0, "primary": 0.0, "multiview": 0.0, "prototype": 0.0,
            "steps": 0, "skipped": 0, "multiview_rows": 0, "prototype_rows": 0,
            "x1": 0, "x5": 0, "xn": 0}
    global_step = int(global_step_start)
    for iteration, batch in enumerate(loader, 1):
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        output = model(data)
        loss_primary = primary_loss(output["logits"], data["patient_id"])
        loss_multiview, n_multiview = multiview_auxiliary_loss(
            model, output["q"], data["positive_cxr_feats"],
            data["positive_cxr_mask"], data["patient_id"])
        loss_prototype, n_prototype = prototype_auxiliary_loss(
            model, output["q"], prototype_raw, data["disease_targets"],
            data["disease_observed"])
        loss = (loss_primary + LAMBDA_MULTIVIEW * loss_multiview
                + LAMBDA_PROTOTYPE * loss_prototype)
        if not torch.isfinite(loss):
            sums["skipped"] += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       max_grad_norm)
        optimizer.step()
        metrics = batch_retrieval_metrics(output["logits"].detach(), data["patient_id"],
                                          c2_rows=data["c2_row"])
        sums["loss"] += float(loss.detach())
        sums["primary"] += float(loss_primary.detach())
        sums["multiview"] += float(loss_multiview.detach())
        sums["prototype"] += float(loss_prototype.detach())
        sums["multiview_rows"] += n_multiview
        sums["prototype_rows"] += n_prototype
        sums["x1"] += metrics["cross_patient_top1_correct"]
        sums["x5"] += metrics["cross_patient_top5_correct"]
        sums["xn"] += metrics["cross_patient_rows"]
        sums["steps"] += 1
        global_step += 1
        if iteration_records is not None and global_step % max(1, dynamics_log_every) == 0:
            iteration_records.append({
                "global_step": global_step, "epoch": epoch, "iter_in_epoch": iteration,
                "loss": float(loss.detach()),
                "cross_patient_loss": float(loss_primary.detach()),
                "multiview_loss": float(loss_multiview.detach()),
                "prototype_loss": float(loss_prototype.detach()),
                "temporal_loss": 0.0, "multiview_rows": n_multiview,
                "prototype_rows": n_prototype,
                "train_cross_patient_R@1": metrics["cross_patient_top1_correct"] / max(metrics["cross_patient_rows"], 1),
                "train_cross_patient_R@5": metrics["cross_patient_top5_correct"] / max(metrics["cross_patient_rows"], 1),
                "train_cross_patient_rows": metrics["cross_patient_rows"],
                "train_temporal_R@1": float("nan"), "train_temporal_R@5": float("nan"),
                "train_temporal_rows": 0,
            })
    steps = max(sums["steps"], 1)
    print(f"      auxiliary: multiview={sums['multiview']/steps:.4f} "
          f"prototype={sums['prototype']/steps:.4f} "
          f"rows/view={sums['multiview_rows']/steps:.1f} "
          f"rows/proto={sums['prototype_rows']/steps:.1f}", flush=True)
    return {
        "loss": sums["loss"] / steps,
        "cross_patient_loss": sums["primary"] / steps,
        "multiview_loss": sums["multiview"] / steps,
        "prototype_loss": sums["prototype"] / steps,
        "lambda_multiview": LAMBDA_MULTIVIEW, "lambda_prototype": LAMBDA_PROTOTYPE,
        "temporal_loss": 0.0, "steps": sums["steps"], "skipped": sums["skipped"],
        "avg_temporal_rows": 0.0,
        "avg_multiview_rows": sums["multiview_rows"] / steps,
        "avg_prototype_rows": sums["prototype_rows"] / steps,
        "cross_patient_batch_top1": sums["x1"] / max(sums["xn"], 1),
        "cross_patient_batch_top5": sums["x5"] / max(sums["xn"], 1),
        "temporal_batch_top1": float("nan"), "temporal_batch_top5": float("nan"),
        "cross_patient_batch_rows": sums["xn"], "temporal_batch_rows": 0,
        "last_global_step": global_step,
    }


def load_label_table(path):
    frame = pd.read_csv(path, usecols=["subject_id", "study_id", *LABEL_COLUMNS])
    keys = zip(frame["subject_id"].astype(int), frame["study_id"].astype(str))
    values = frame[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    return {key: values[index] for index, key in enumerate(keys)}


def attach_labels_and_prototypes(data, args, labels_csv):
    pair_json = os.environ.get("PAIR_JSON") or args.seq_target_pairs or args.single_pairs
    if not pair_json:
        raise ValueError("PAIR_JSON, --seq_target_pairs, or --single_pairs is required")
    raw = json.loads(Path(pair_json).read_text())["pairs"]
    cxr_index = {str(value): i for i, value in enumerate(json.loads(Path(args.cxr_ids).read_text()))}
    label_table = load_label_table(labels_csv)
    positive_rows = {}
    labels_by_primary = {}
    prototype_sums = np.zeros((len(LABEL_COLUMNS), data.cxr_emb.shape[1]), np.float64)
    prototype_counts = np.zeros(len(LABEL_COLUMNS), np.int64)
    seen_train_studies = set()
    for row in raw:
        primary_id = row.get("cxr_t2", row.get("cxr_id"))
        primary = cxr_index.get(str(primary_id))
        if primary is None:
            continue
        views = [cxr_index[str(value)] for value in row["cxr_positive_ids"]
                 if str(value) in cxr_index]
        positive_rows[primary] = views or [primary]
        values = label_table.get((int(row["patient_id"]), str(row["study_id"])))
        if values is None:
            values = np.full(len(LABEL_COLUMNS), np.nan, np.float32)
        observed = np.isin(values, [0.0, 1.0]).astype(np.float32)
        targets = np.where(values == 1.0, 1.0, 0.0).astype(np.float32)
        labels_by_primary[primary] = (targets, observed)
        study_key = (int(row["patient_id"]), str(row["study_id"]))
        if (data.patient_to_split.get(int(row["patient_id"])) == "train"
                and study_key not in seen_train_studies):
            seen_train_studies.add(study_key)
            study_embedding = data.cxr_emb[positive_rows[primary]].mean(0)
            for label_index in np.flatnonzero(targets > 0.5):
                prototype_sums[label_index] += study_embedding
                prototype_counts[label_index] += 1
    if np.any(prototype_counts == 0):
        raise RuntimeError(f"Empty disease prototype: counts={prototype_counts.tolist()}")
    prototypes = (prototype_sums / prototype_counts[:, None]).astype(np.float32)
    data.positive_rows_by_primary = positive_rows
    data.labels_by_primary = labels_by_primary
    data.disease_prototypes = torch.from_numpy(prototypes)
    print("  Prototype train-study counts: "
          + ", ".join(f"{name}={count:,}" for name, count in zip(LABEL_COLUMNS, prototype_counts)),
          flush=True)


def main():
    public = Path(os.environ.get("MIMIC_CXR_ROOT", ROOT / "data" / "mimic-cxr-jpg"))
    labels_csv = Path(os.environ.get(
        "LABELS_CSV", public / "mimic-cxr-2.0.0-chexpert.csv.gz"))
    original_resolve = runner.resolve_specs
    original_load = engine.load_staged_data

    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one sequence experiment")
        spec = specs[0]
        spec.name = os.environ.get(
            "EXPERIMENT_NAME", "latest_gated_primary_multiview_prototype_0_24h")
        spec.description = os.environ.get(
            "EXPERIMENT_DESCRIPTION",
            f"Primary InfoNCE + {LAMBDA_MULTIVIEW:g} multiview + "
            f"{LAMBDA_PROTOTYPE:g} six-disease prototype")
        spec.target_window = os.environ.get(
            "TARGET_WINDOW", "[t2-24h,t2], pooled multiview")
        spec.ecg_pool, spec.use_future_query = "mean", False
        spec.use_time_embedding = True
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        spec.temporal_min_horizon_hours = None
        spec.temporal_max_horizon_hours = None
        return specs

    def load_data(spec, args):
        data = original_load(spec, args)
        attach_labels_and_prototypes(data, args, labels_csv)
        return data

    runner.resolve_specs = resolve
    engine.load_staged_data = load_data
    engine.StagedModel = LatestGatedHistoryModel
    engine.StagedDataset = PrototypeDataset
    engine.collate_fn = prototype_collate
    engine._run_epoch = run_epoch
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve(args.only)[0]
    data = load_data(spec, args)
    engine.fit(spec, args, data=data, device=device)


if __name__ == "__main__":
    main()
