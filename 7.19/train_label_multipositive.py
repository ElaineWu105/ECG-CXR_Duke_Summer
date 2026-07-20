#!/usr/bin/env python3
"""Train ECG->CXR contrastive learning with label-guided multi-positives.

The model architecture and retrieval evaluation are unchanged.  The training
objective is

    L = L_instance_InfoNCE + lambda_label * L_label_multi_positive

where off-diagonal CXR candidates from different patients become additional
soft positives when they share one or more explicitly positive CheXpert labels.
Uncertain (-1) and missing labels never create positives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(EXP))

import engine  # noqa: E402
import run_experiments as runner  # noqa: E402
from losses import batch_retrieval_metrics, total_loss  # noqa: E402
from staged_dataset import (  # noqa: E402
    StagedDataset as BaseStagedDataset,
    collate_fn as base_collate_fn,
)


LABEL_COLUMNS = (
    "Pneumonia",
    "Consolidation",
    "Lung Opacity",
    "Pneumothorax",
    "Pleural Effusion",
    "Edema",
)


def load_cxr_labels(labels_csv: Path, metadata_csv: Path, cxr_ids_json: Path):
    """Return row-aligned binary targets and observed masks for CXR embeddings."""
    labels = pd.read_csv(
        labels_csv, usecols=["subject_id", "study_id", *LABEL_COLUMNS]
    )
    metadata = pd.read_csv(
        metadata_csv, usecols=["dicom_id", "subject_id", "study_id"]
    )
    merged = metadata.merge(
        labels, on=["subject_id", "study_id"], how="left", validate="many_to_one"
    )
    by_dicom = merged.set_index(merged["dicom_id"].astype(str))
    cxr_ids = [str(x) for x in json.loads(cxr_ids_json.read_text())]

    targets = np.zeros((len(cxr_ids), len(LABEL_COLUMNS)), dtype=np.float32)
    observed = np.zeros_like(targets)
    matched = 0
    for row, dicom_id in enumerate(cxr_ids):
        if dicom_id not in by_dicom.index:
            continue
        values = by_dicom.loc[dicom_id, list(LABEL_COLUMNS)]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[0]
        values = values.to_numpy(dtype=np.float32)
        mask = np.isin(values, [0.0, 1.0])
        targets[row] = np.where(mask, values, 0.0)
        observed[row] = mask.astype(np.float32)
        matched += 1
    print(f"  CXR labels: matched {matched:,}/{len(cxr_ids):,} embedding rows")
    return targets, observed


class LabelStagedDataset(BaseStagedDataset):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        row = int(item["c2_row"])
        item["cxr_labels"] = torch.from_numpy(self.data.cxr_labels[row])
        item["cxr_label_mask"] = torch.from_numpy(self.data.cxr_label_mask[row])
        return item


def label_collate_fn(batch):
    out = base_collate_fn(batch)
    out["cxr_labels"] = torch.stack([x["cxr_labels"] for x in batch])
    out["cxr_label_mask"] = torch.stack([x["cxr_label_mask"] for x in batch])
    return out


def label_multi_positive_loss(logits, patient_ids, labels, observed, top_k=0):
    """Soft multi-positive CE using Jaccard overlap of explicit positives.

    The diagonal remains a positive with weight 1. Off-diagonal pairs are
    positives only if they are from different patients and share an explicitly
    positive label. Their target weight is positive-label Jaccard similarity.
    """
    batch_size = logits.size(0)
    same_patient = patient_ids[:, None].eq(patient_ids[None, :])
    eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    valid = (~same_patient) | eye

    positive = (labels > 0.5) & (observed > 0.5)
    positive_f = positive.float()
    intersection = positive_f @ positive_f.t()
    positive_count = positive_f.sum(dim=1)
    union = positive_count[:, None] + positive_count[None, :] - intersection
    similarity = torch.where(union > 0, intersection / union.clamp_min(1.0), union)
    similarity = similarity.masked_fill(same_patient & ~eye, 0.0)
    similarity.fill_diagonal_(1.0)
    if top_k and top_k < batch_size - 1:
        off_diagonal = similarity.masked_fill(eye, 0.0)
        keep_indices = off_diagonal.topk(int(top_k), dim=1).indices
        keep = torch.zeros_like(similarity, dtype=torch.bool)
        keep.scatter_(1, keep_indices, True)
        similarity = similarity.masked_fill(~(keep | eye), 0.0)


    has_extra_positive = ((similarity > 0) & ~eye).any(dim=1)
    if not has_extra_positive.any():
        return logits.new_zeros(()), 0, 0.0

    target = similarity / similarity.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = F.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=1)
    # Avoid 0 * -inf producing NaNs.
    per_row = -(torch.where(target > 0, target * log_prob, torch.zeros_like(log_prob))).sum(1)
    loss = per_row[has_extra_positive].mean()
    extra = ((similarity > 0) & ~eye).sum(dim=1).float()
    return loss, int(has_extra_positive.sum()), float(extra[has_extra_positive].mean())


def make_run_epoch(lambda_label: float, label_top_k: int):
    def run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm,
                  temporal_min_horizon_hours=None, temporal_max_horizon_hours=None,
                  epoch=0, global_step_start=0, iteration_records=None,
                  dynamics_log_every=1):
        model.train()
        sums = {"loss": 0.0, "cross_patient_loss": 0.0, "temporal_loss": 0.0,
                "label_multi_positive_loss": 0.0, "steps": 0, "skipped": 0,
                "n_temporal_rows": 0, "label_rows": 0, "extra_positives": 0.0,
                "x1": 0, "x5": 0, "xn": 0, "t1": 0, "t5": 0, "tn": 0}
        global_step = int(global_step_start)
        for iter_in_epoch, batch in enumerate(loader, start=1):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(b)
            base, logs = total_loss(
                out["logits"], b["patient_id"], w_cross, w_temporal,
                c2_rows=b.get("c2_row"), c2_times_h=b.get("c2_time_h"),
                ecg_times_h=b.get("ecg_times_h"), ecg_mask=b.get("ecg_mask"),
                temporal_min_horizon_hours=temporal_min_horizon_hours,
                temporal_max_horizon_hours=temporal_max_horizon_hours,
            )
            label_loss, n_label_rows, mean_extra = label_multi_positive_loss(
                out["logits"], b["patient_id"], b["cxr_labels"], b["cxr_label_mask"], top_k=label_top_k
            )
            loss = base + float(lambda_label) * label_loss
            if not torch.isfinite(loss):
                sums["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()

            metrics = batch_retrieval_metrics(
                out["logits"].detach(), b["patient_id"], c2_rows=b.get("c2_row"),
                c2_times_h=b.get("c2_time_h"), ecg_times_h=b.get("ecg_times_h"),
                ecg_mask=b.get("ecg_mask"),
                temporal_min_horizon_hours=temporal_min_horizon_hours,
                temporal_max_horizon_hours=temporal_max_horizon_hours,
            )
            sums["loss"] += float(loss.detach())
            sums["cross_patient_loss"] += logs["cross_patient_loss"]
            sums["temporal_loss"] += logs["temporal_loss"]
            sums["label_multi_positive_loss"] += float(label_loss.detach())
            sums["n_temporal_rows"] += logs["n_temporal_rows"]
            sums["label_rows"] += n_label_rows
            sums["extra_positives"] += mean_extra
            sums["x1"] += metrics["cross_patient_top1_correct"]
            sums["x5"] += metrics["cross_patient_top5_correct"]
            sums["xn"] += metrics["cross_patient_rows"]
            sums["t1"] += metrics["temporal_top1_correct"]
            sums["t5"] += metrics["temporal_top5_correct"]
            sums["tn"] += metrics["temporal_rows"]
            sums["steps"] += 1
            global_step += 1

            if iteration_records is not None and global_step % max(1, dynamics_log_every) == 0:
                iteration_records.append({
                    "global_step": global_step, "epoch": epoch,
                    "iter_in_epoch": iter_in_epoch, "loss": float(loss.detach()),
                    "cross_patient_loss": logs["cross_patient_loss"],
                    "temporal_loss": logs["temporal_loss"],
                    "label_multi_positive_loss": float(label_loss.detach()),
                    "label_rows": n_label_rows,
                    "mean_extra_positives": mean_extra,
                    "train_cross_patient_R@1": metrics["cross_patient_top1_correct"] / max(metrics["cross_patient_rows"], 1),
                    "train_cross_patient_R@5": metrics["cross_patient_top5_correct"] / max(metrics["cross_patient_rows"], 1),
                    "train_cross_patient_rows": metrics["cross_patient_rows"],
                    "train_temporal_R@1": metrics["temporal_top1_correct"] / max(metrics["temporal_rows"], 1),
                    "train_temporal_R@5": metrics["temporal_top5_correct"] / max(metrics["temporal_rows"], 1),
                    "train_temporal_rows": metrics["temporal_rows"],
                })

        steps = max(sums["steps"], 1)
        ratio = lambda a, b: a / b if b else float("nan")
        return {
            "loss": sums["loss"] / steps,
            "cross_patient_loss": sums["cross_patient_loss"] / steps,
            "temporal_loss": sums["temporal_loss"] / steps,
            "label_multi_positive_loss": sums["label_multi_positive_loss"] / steps,
            "lambda_label": float(lambda_label), "label_top_k": int(label_top_k), "steps": sums["steps"],
            "skipped": sums["skipped"],
            "avg_temporal_rows": sums["n_temporal_rows"] / steps,
            "avg_label_rows": sums["label_rows"] / steps,
            "avg_extra_positives": sums["extra_positives"] / steps,
            "cross_patient_batch_top1": ratio(sums["x1"], sums["xn"]),
            "cross_patient_batch_top5": ratio(sums["x5"], sums["xn"]),
            "temporal_batch_top1": ratio(sums["t1"], sums["tn"]),
            "temporal_batch_top5": ratio(sums["t5"], sums["tn"]),
            "cross_patient_batch_rows": sums["xn"],
            "temporal_batch_rows": sums["tn"], "last_global_step": global_step,
        }
    return run_epoch


def main():
    public_cxr = Path(os.environ.get("MIMIC_CXR_ROOT", ROOT / "data" / "mimic-cxr-jpg"))
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--lambda_label", type=float, default=0.5)
    extra.add_argument("--label_top_k", type=int, default=0)
    extra.add_argument("--labels_csv", type=Path,
                       default=public_cxr / "mimic-cxr-2.0.0-chexpert.csv.gz")
    extra.add_argument("--metadata_csv", type=Path,
                       default=public_cxr / "mimic-cxr-2.0.0-metadata.csv.gz")
    custom, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    original_resolve = runner.resolve_specs
    original_load = engine.load_staged_data

    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one baseline experiment")
        spec = specs[0]
        spec.name = os.environ.get("MULTIPOS_NAME", "single_n2_label_multipos")
        spec.description = "Single ECG with instance InfoNCE + label multi-positive loss"
        spec.target_window = os.environ.get("TARGET_WINDOW", "[t2-14h, t2-2h]")
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        return specs

    def load_data(spec, args):
        data = original_load(spec, args)
        targets, masks = load_cxr_labels(
            custom.labels_csv, custom.metadata_csv, Path(args.cxr_ids)
        )
        data.cxr_labels = targets
        data.cxr_label_mask = masks
        return data

    runner.resolve_specs = resolve
    runner.load_staged_data = load_data
    engine.load_staged_data = load_data
    engine.StagedDataset = LabelStagedDataset
    engine.collate_fn = label_collate_fn
    engine._run_epoch = make_run_epoch(custom.lambda_label, custom.label_top_k)
    runner.main()


if __name__ == "__main__":
    main()
