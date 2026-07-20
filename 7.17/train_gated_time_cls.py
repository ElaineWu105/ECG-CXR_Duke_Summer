#!/usr/bin/env python3
"""Train CLS pooling with a learnable scalar gate on ECG-to-t2 time embeddings."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "7.15"))
sys.path.insert(2, str(EXP))

import engine  # noqa: E402
import run_experiments as runner  # noqa: E402
from train_cls_diagnostics import DiagnosticStagedModel, diagnostic_epoch  # noqa: E402


class GatedTimeCLSModel(DiagnosticStagedModel):
    """Use h_ecg + sigmoid(gate_logit) * h_time before the Transformer."""

    def __init__(self, *args, time_gate_init_logit=-3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_gate_logit = nn.Parameter(torch.tensor(float(time_gate_init_logit)))

    def reset_cls_diagnostics(self):
        super().reset_cls_diagnostics()
        self._diag.update({
            "token_stat_batches": 0,
            "ecg_token_norm_sum": 0.0,
            "raw_time_token_norm_sum": 0.0,
            "gated_time_token_norm_sum": 0.0,
        })

    def _encode_sequence_tokens(self, batch, add_pool_token=False):
        feats = batch["ecg_feats"]
        mask = batch["ecg_mask"].bool()
        if self.spec.ecg_perturb == "zero":
            h = feats.new_zeros(feats.size(0), feats.size(1), self.d_model)
            return h, mask, False

        batch_size = feats.size(0)
        ecg_h = self.ecg_in_proj(feats)
        h = ecg_h
        if self.seq_time_emb is not None:
            time_h = self.seq_time_emb(batch["ecg_t2t"])
            gate = torch.sigmoid(self.time_gate_logit)
            h = ecg_h + gate * time_h
            if self.training:
                valid = mask.unsqueeze(-1).expand_as(ecg_h)
                ecg_norm = ecg_h.masked_select(valid).reshape(-1, self.d_model).norm(dim=-1).mean()
                time_norm = time_h.masked_select(valid).reshape(-1, self.d_model).norm(dim=-1).mean()
                self._diag["token_stat_batches"] += 1
                self._diag["ecg_token_norm_sum"] += float(ecg_norm.detach())
                self._diag["raw_time_token_norm_sum"] += float(time_norm.detach())
                self._diag["gated_time_token_norm_sum"] += float((gate.detach() * time_norm).detach())

        h, mask = self._apply_sequence_token_dropout(h, mask)
        has_pool_token = False
        if add_pool_token and self.future_query is not None:
            q_tok = self.future_query.expand(batch_size, 1, -1)
            if self.future_time_emb is not None:
                q_tok = q_tok + self.future_time_emb(batch["delta_t"]).unsqueeze(1)
            h = torch.cat([q_tok, h], dim=1)
            pad = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)
            has_pool_token = True
        elif add_pool_token and self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, 1, -1)
            h = torch.cat([cls, h], dim=1)
            pad = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([pad, mask], dim=1)
            has_pool_token = True

        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.enc_norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h, mask, has_pool_token

    def cls_diagnostic_summary(self):
        result = super().cls_diagnostic_summary()
        count = max(int(self._diag["token_stat_batches"]), 1)
        ecg_norm = self._diag["ecg_token_norm_sum"] / count
        raw_time_norm = self._diag["raw_time_token_norm_sum"] / count
        gated_time_norm = self._diag["gated_time_token_norm_sum"] / count
        result.update({
            "time_gate_logit": float(self.time_gate_logit.detach()),
            "time_gate_value": float(torch.sigmoid(self.time_gate_logit.detach())),
            "ecg_token_norm": ecg_norm,
            "raw_time_token_norm": raw_time_norm,
            "gated_time_token_norm": gated_time_norm,
            "gated_time_to_ecg_norm_ratio": gated_time_norm / (ecg_norm + 1e-12),
        })
        return result


def write_gate_history(history, path):
    rows = []
    for item in history:
        diag = item["train"].get("cls_diagnostics", {})
        cp = item["val"].get("cross_patient", {})
        rows.append({
            "epoch": item["epoch"],
            "train_loss": item["train"].get("loss"),
            "train_batch_recall@1": item["train"].get("cross_patient_batch_top1"),
            "val_recall@1": cp.get("recall@1"),
            "val_recall@5": cp.get("recall@5"),
            "val_mrr": cp.get("mrr"),
            **diag,
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    n_value = int(os.environ.get("N_VALUE", "2"))
    gate_init = float(os.environ.get("TIME_GATE_INIT_LOGIT", "-3.0"))
    original_resolve = runner.resolve_specs

    def resolve_gated(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one sequence experiment")
        spec = specs[0]
        spec.name = f"case2_gated_time_cls_n{n_value}"
        spec.description = "ECG sequence CLS with a learned scalar time-embedding gate"
        spec.target_window = f"[t2-{n_value + 12}h, t2-{n_value}h]"
        spec.ecg_pool = "cls"
        spec.use_future_query = False
        spec.loss_mode = "cross"
        spec.lambda_temporal = 0.0
        return specs

    class ConfiguredGatedModel(GatedTimeCLSModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, time_gate_init_logit=gate_init, **kwargs)

    runner.resolve_specs = resolve_gated
    engine.StagedModel = ConfiguredGatedModel
    engine._run_epoch = diagnostic_epoch(engine._run_epoch)
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve_gated(args.only)[0]
    data = engine.load_staged_data(spec, args)
    result = engine.fit(spec, args, data=data, device=device)
    run_dir = Path(args.output_dir) / spec.name
    write_gate_history(result["history"], run_dir / "gate_diagnostics.csv")
    print(f"Gate diagnostics: {run_dir / 'gate_diagnostics.csv'}")


if __name__ == "__main__":
    main()
