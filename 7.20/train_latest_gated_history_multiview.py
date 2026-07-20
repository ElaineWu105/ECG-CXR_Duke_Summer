#!/usr/bin/env python3
"""Latest gated-history model with same-study CXR multi-positive InfoNCE."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "7.15"))
sys.path.insert(2, str(EXP))

import engine  # noqa: E402
import run_experiments as runner  # noqa: E402
from losses import batch_retrieval_metrics  # noqa: E402
from staged_dataset import StagedDataset as BaseDataset, collate_fn as base_collate  # noqa: E402
from train_latest_gated_history import LatestGatedHistoryModel  # noqa: E402


class MultiViewDataset(BaseDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        rows = self.data.positive_rows_by_primary[int(item["c2_row"])]
        item["positive_cxr_feats"] = torch.from_numpy(
            self.data.cxr_emb[rows].astype(np.float32))
        return item


def multiview_collate(batch):
    output = base_collate(batch)
    batch_size = len(batch)
    max_views = max(x["positive_cxr_feats"].shape[0] for x in batch)
    feature_dim = batch[0]["positive_cxr_feats"].shape[1]
    feats = torch.zeros(batch_size, max_views, feature_dim)
    mask = torch.zeros(batch_size, max_views, dtype=torch.bool)
    for i, item in enumerate(batch):
        count = item["positive_cxr_feats"].shape[0]
        feats[i, :count] = item["positive_cxr_feats"]
        mask[i, :count] = True
    output["positive_cxr_feats"] = feats
    output["positive_cxr_mask"] = mask
    return output


def multi_positive_cross_loss(model, query, positive_feats, positive_mask, patient_ids):
    batch_size, n_views, feature_dim = positive_feats.shape
    projected = model.cxr_proj(positive_feats.reshape(-1, feature_dim)).reshape(
        batch_size, n_views, -1)
    candidates = projected.reshape(batch_size * n_views, -1)
    candidate_valid = positive_mask.reshape(-1)
    owners = torch.arange(batch_size, device=query.device).repeat_interleave(n_views)
    candidate_patients = patient_ids[owners]
    logits = (query @ candidates.t()) * torch.exp(model.logit_scale).clamp(max=100.0)
    same_patient = patient_ids[:, None].eq(candidate_patients[None, :])
    own_study = owners[None, :].eq(torch.arange(batch_size, device=query.device)[:, None])
    valid = candidate_valid[None, :] & ((~same_patient) | own_study)
    positive = candidate_valid[None, :] & own_study
    log_den = torch.logsumexp(logits.masked_fill(~valid, float("-inf")), dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    return -(log_num - log_den).mean()


def run_epoch(model, loader, optimizer, device, w_cross, w_temporal, max_grad_norm,
              temporal_min_horizon_hours=None, temporal_max_horizon_hours=None,
              epoch=0, global_step_start=0, iteration_records=None, dynamics_log_every=1):
    del w_cross, w_temporal, temporal_min_horizon_hours, temporal_max_horizon_hours
    model.train()
    sums = {"loss": 0.0, "steps": 0, "skipped": 0, "views": 0.0,
            "x1": 0, "x5": 0, "xn": 0}
    global_step = int(global_step_start)
    for iteration, batch in enumerate(loader, 1):
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        output = model(data)
        loss = multi_positive_cross_loss(
            model, output["q"], data["positive_cxr_feats"],
            data["positive_cxr_mask"], data["patient_id"])
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
        sums["views"] += float(data["positive_cxr_mask"].sum(1).float().mean())
        sums["x1"] += metrics["cross_patient_top1_correct"]
        sums["x5"] += metrics["cross_patient_top5_correct"]
        sums["xn"] += metrics["cross_patient_rows"]
        sums["steps"] += 1
        global_step += 1
        if iteration_records is not None and global_step % max(1, dynamics_log_every) == 0:
            iteration_records.append({
                "global_step": global_step, "epoch": epoch, "iter_in_epoch": iteration,
                "loss": float(loss.detach()), "cross_patient_loss": float(loss.detach()),
                "temporal_loss": 0.0,
                "mean_positive_views": float(data["positive_cxr_mask"].sum(1).float().mean()),
                "train_cross_patient_R@1": metrics["cross_patient_top1_correct"] / max(metrics["cross_patient_rows"], 1),
                "train_cross_patient_R@5": metrics["cross_patient_top5_correct"] / max(metrics["cross_patient_rows"], 1),
                "train_cross_patient_rows": metrics["cross_patient_rows"],
                "train_temporal_R@1": float("nan"), "train_temporal_R@5": float("nan"),
                "train_temporal_rows": 0,
            })
    steps = max(sums["steps"], 1)
    return {
        "loss": sums["loss"] / steps, "cross_patient_loss": sums["loss"] / steps,
        "temporal_loss": 0.0, "steps": sums["steps"], "skipped": sums["skipped"],
        "avg_temporal_rows": 0.0, "mean_positive_views": sums["views"] / steps,
        "cross_patient_batch_top1": sums["x1"] / max(sums["xn"], 1),
        "cross_patient_batch_top5": sums["x5"] / max(sums["xn"], 1),
        "temporal_batch_top1": float("nan"), "temporal_batch_top5": float("nan"),
        "cross_patient_batch_rows": sums["xn"], "temporal_batch_rows": 0,
        "last_global_step": global_step,
    }


def main():
    original_resolve = runner.resolve_specs
    original_load = engine.load_staged_data

    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one sequence experiment")
        spec = specs[0]
        spec.name = os.environ.get("EXPERIMENT_NAME", "latest_gated_history_multiview_0_24h")
        spec.description = "Latest gated history with same-study AP/PA/LATERAL/LL positives"
        spec.target_window = "[t2-24h,t2], pooled; same-study multiview positives"
        spec.ecg_pool, spec.use_future_query = "mean", False
        spec.use_time_embedding = True
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        spec.temporal_min_horizon_hours = None
        spec.temporal_max_horizon_hours = None
        return specs

    def load_data(spec, args):
        data = original_load(spec, args)
        raw = json.loads(Path(args.seq_target_pairs).read_text())["pairs"]
        cxr_index = {str(value): i for i, value in enumerate(json.loads(Path(args.cxr_ids).read_text()))}
        mapping = {}
        for row in raw:
            primary = cxr_index.get(str(row["cxr_t2"]))
            positives = [cxr_index[x] for x in map(str, row["cxr_positive_ids"]) if x in cxr_index]
            if primary is not None:
                mapping[primary] = positives or [primary]
        data.positive_rows_by_primary = mapping
        return data

    runner.resolve_specs = resolve
    engine.load_staged_data = load_data
    engine.StagedModel = LatestGatedHistoryModel
    engine.StagedDataset = MultiViewDataset
    engine.collate_fn = multiview_collate
    engine._run_epoch = run_epoch
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve(args.only)[0]
    data = load_data(spec, args)
    engine.fit(spec, args, data=data, device=device)


if __name__ == "__main__":
    main()
