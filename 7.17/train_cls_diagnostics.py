#!/usr/bin/env python3
"""Train n=2 ECG-sequence CLS and diagnose whether CLS learns ECG information."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(ROOT / "7.15"))
sys.path.insert(1, str(EXP))

import config as C  # noqa: E402
import engine  # noqa: E402
import metrics  # noqa: E402
import run_experiments as runner  # noqa: E402
from staged_dataset import StagedDataset, collate_fn  # noqa: E402
from staged_model import StagedModel as BaseStagedModel  # noqa: E402


class DiagnosticStagedModel(BaseStagedModel):
    """Original model plus training-only CLS optimization/representation stats."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cls_token is None:
            raise ValueError("CLS diagnostics require ecg_pool='cls'")
        self.register_buffer("_cls_initial", self.cls_token.detach().clone(), persistent=False)
        self.cls_token.register_hook(self._record_cls_grad)
        self.reset_cls_diagnostics()

    def reset_cls_diagnostics(self):
        self._diag = {
            "batches": 0, "cls_output_std_sum": 0.0,
            "cls_output_pair_cos_sum": 0.0, "similarity_margin_sum": 0.0,
            "grad_steps": 0, "cls_grad_norm_sum": 0.0, "cls_grad_norm_max": 0.0,
        }

    def _record_cls_grad(self, grad):
        norm = float(grad.detach().norm())
        self._diag["grad_steps"] += 1
        self._diag["cls_grad_norm_sum"] += norm
        self._diag["cls_grad_norm_max"] = max(self._diag["cls_grad_norm_max"], norm)
        return grad

    def _encode_sequence(self, batch):
        feats = batch["ecg_feats"]
        if self.spec.ecg_perturb == "zero":
            return feats.new_zeros(feats.size(0), self.ecg_out_dim)
        h, mask, has_pool_token = self._encode_sequence_tokens(batch, add_pool_token=True)
        pooled = h[:, 0] if has_pool_token else self._masked_mean(h, mask)
        if self.training:
            z = F.normalize(pooled.detach().float(), dim=-1)
            self._diag["batches"] += 1
            self._diag["cls_output_std_sum"] += float(pooled.detach().float().std(dim=0).mean())
            if z.size(0) > 1:
                cos = z @ z.t()
                off = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
                self._diag["cls_output_pair_cos_sum"] += float(cos[off].mean())
        if self.seq_pool_norm is not None:
            pooled = self.seq_pool_norm(pooled)
        if self.seq_pool_drop is not None:
            pooled = self.seq_pool_drop(pooled)
        return pooled

    def forward(self, batch):
        out = super().forward(batch)
        if self.training and out["q"].size(0) > 1:
            sims = out["q"].detach().float() @ out["c2"].detach().float().t()
            off = ~torch.eye(sims.size(0), dtype=torch.bool, device=sims.device)
            self._diag["similarity_margin_sum"] += float(sims.diag().mean() - sims[off].mean())
        return out

    def cls_diagnostic_summary(self):
        batches = max(self._diag["batches"], 1)
        grad_steps = max(self._diag["grad_steps"], 1)
        initial, current = self._cls_initial.detach(), self.cls_token.detach()
        drift = float((current - initial).norm())
        return {
            "cls_param_norm": float(current.norm()),
            "cls_param_drift": drift,
            "cls_relative_drift": drift / (float(initial.norm()) + 1e-12),
            "cls_grad_norm_mean": self._diag["cls_grad_norm_sum"] / grad_steps,
            "cls_grad_norm_max": self._diag["cls_grad_norm_max"],
            "cls_output_feature_std": self._diag["cls_output_std_sum"] / batches,
            "cls_output_between_patient_cosine": self._diag["cls_output_pair_cos_sum"] / batches,
            "positive_negative_cosine_margin": self._diag["similarity_margin_sum"] / batches,
        }


class BatchPerturbationModel:
    """Read-only view that changes ECG features/times immediately before encode."""

    def __init__(self, model, mode):
        self.model, self.mode, self.cxr_proj = model, mode, model.cxr_proj

    def eval(self):
        self.model.eval()
        return self

    def encode(self, batch):
        b = dict(batch)
        if self.mode == "zero_ecg":
            b["ecg_feats"] = torch.zeros_like(batch["ecg_feats"])
        elif self.mode == "shuffle_time":
            times = batch["ecg_t2t"].clone()
            for i, length in enumerate(batch["ecg_mask"].sum(dim=1).tolist()):
                length = int(length)
                if length > 1:
                    times[i, :length] = times[i, :length].flip(0)
            b["ecg_t2t"] = times
        return self.model.encode(b)


def diagnostic_epoch(original):
    def run(model, *args, **kwargs):
        model.reset_cls_diagnostics()
        result = original(model, *args, **kwargs)
        result["cls_diagnostics"] = model.cls_diagnostic_summary()
        return result
    return run


def build_model(spec, data, args, device, checkpoint):
    model = DiagnosticStagedModel(
        spec, cxr_dim=data.cxr_emb.shape[1], ecg_dim=data.ecg_emb.shape[1],
        proj_dim=args.proj_dim, cxr_proj_hidden=C.CXR_PROJ_HIDDEN,
        d_model=args.d_model, ecg_tx_layers=args.ecg_tx_layers,
        ecg_tx_heads=C.ECG_TX_HEADS, ecg_tx_mlp_ratio=C.ECG_TX_MLP_RATIO,
        fusion_hidden=C.FUSION_HIDDEN, time_emb_dim=C.TIME_EMB_DIM,
        dropout=C.DROPOUT, temperature=args.temperature,
        learnable_temperature=args.learnable_temperature).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    model.eval()
    return model


def ablation_eval(model, data, split, device, batch_size, seed):
    indices = data.split_indices[split]
    normal = StagedDataset(data, indices, ecg_perturb="none", seed=seed)
    shuffled = StagedDataset(data, indices, ecg_perturb="shuffle", seed=seed)
    evaluate = lambda m, ds: metrics.evaluate_retrieval(
        m, ds, data.cxr_emb, device, batch_size, collate_fn)
    return {
        "normal": evaluate(model, normal),
        "zero_ecg": evaluate(BatchPerturbationModel(model, "zero_ecg"), normal),
        "shuffled_patient_ecg": evaluate(model, shuffled),
        "shuffled_ecg_time": evaluate(BatchPerturbationModel(model, "shuffle_time"), normal),
    }


def write_cls_csv(history, path):
    rows = []
    for item in history:
        row = {"epoch": item["epoch"], **item["train"].get("cls_diagnostics", {})}
        cp = item["val"].get("cross_patient", {})
        row.update({
            "train_loss": item["train"].get("loss"),
            "train_batch_recall@1": item["train"].get("cross_patient_batch_top1"),
            "val_recall@1": cp.get("recall@1"), "val_recall@5": cp.get("recall@5"),
            "val_mrr": cp.get("mrr"),
        })
        rows.append(row)
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_ablation_csv(ablations, path):
    rows = []
    for split, variants in ablations.items():
        for variant, result in variants.items():
            rows.append({"split": split, "variant": variant,
                         **result.get("cross_patient", {})})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    original_resolve = runner.resolve_specs

    def resolve_cls(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one sequence experiment")
        spec = specs[0]
        spec.name = os.environ.get("CLS_DIAG_NAME", "case2_cls_diagnostics_n2")
        spec.description = "ECG sequence CLS pooling with CLS diagnostics"
        spec.target_window = "[t2-14h, t2-2h]"
        spec.ecg_pool, spec.use_future_query = "cls", False
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        return specs

    runner.resolve_specs = resolve_cls
    engine.StagedModel = DiagnosticStagedModel
    engine._run_epoch = diagnostic_epoch(engine._run_epoch)
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve_cls(args.only)[0]
    data = engine.load_staged_data(spec, args)
    result = engine.fit(spec, args, data=data, device=device)

    run_dir = Path(args.output_dir) / spec.name
    model = build_model(spec, data, args, device, run_dir / "best.pt")
    ablations = {
        "validation": ablation_eval(model, data, "val", device, args.eval_batch_size, args.seed + 101),
        "test": ablation_eval(model, data, "test", device, args.eval_batch_size, args.seed + 202),
    }
    result["best_checkpoint_ablations"] = ablations
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    (run_dir / "ablation_results.json").write_text(json.dumps(ablations, indent=2))
    write_cls_csv(result["history"], run_dir / "cls_diagnostics.csv")
    write_ablation_csv(ablations, run_dir / "ablation_results.csv")
    print(f"CLS diagnostics: {run_dir / 'cls_diagnostics.csv'}")
    print(f"Ablations:       {run_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
