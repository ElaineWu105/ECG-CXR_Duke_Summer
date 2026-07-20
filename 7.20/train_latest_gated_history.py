#!/usr/bin/env python3
"""Latest ECG plus gated history-change model for pooled 0-24h sequences."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(ROOT / "7.15"))
sys.path.insert(1, str(EXP))

import engine  # noqa: E402
import run_experiments as runner  # noqa: E402
from staged_model import StagedModel  # noqa: E402


class LatestGatedHistoryModel(StagedModel):
    """Use latest ECG as the state and older ECGs only as a gated correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d_model
        self.current_norm = nn.LayerNorm(d)
        self.history_change = nn.Sequential(
            nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(2 * d, d),
        )
        self.history_score = nn.Linear(d, 1)
        self.history_norm = nn.LayerNorm(d)
        self.history_gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(d)

        # This architecture deliberately replaces generic self-attention with
        # explicit latest/history roles. Do not optimize unused Transformer weights.
        if self.encoder is not None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _encode_sequence(self, batch):
        feats = batch["ecg_feats"]
        mask = batch["ecg_mask"].bool()
        if self.spec.ecg_perturb == "zero":
            return feats.new_zeros(feats.size(0), self.ecg_out_dim)

        tokens = self.ecg_in_proj(feats)
        if self.seq_time_emb is not None:
            tokens = tokens + self.seq_time_emb(batch["ecg_t2t"])

        batch_size, length, _ = tokens.shape
        lengths = mask.sum(dim=1).long().clamp(min=1)
        latest_index = lengths - 1
        row_index = torch.arange(batch_size, device=tokens.device)
        latest = self.current_norm(tokens[row_index, latest_index])

        latest_all = latest.unsqueeze(1).expand(-1, length, -1)
        change_input = torch.cat(
            [tokens, latest_all, latest_all - tokens, latest_all * tokens], dim=-1
        )
        changes = self.history_change(change_input)

        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        history_mask = mask & (positions < latest_index.unsqueeze(1))
        scores = self.history_score(changes).squeeze(-1).masked_fill(~history_mask, -1e9)
        weights = torch.softmax(scores, dim=1) * history_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        history = (changes * weights.unsqueeze(-1)).sum(dim=1)
        history = self.history_norm(history)

        has_history = history_mask.any(dim=1, keepdim=True)
        gate = self.history_gate(torch.cat([latest, history], dim=-1))
        gate = gate * has_history.float()
        pooled = self.output_norm(latest + gate * history)

        if self.seq_pool_drop is not None:
            pooled = self.seq_pool_drop(pooled)
        return pooled


def main():
    original_resolve = runner.resolve_specs

    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("Select exactly one sequence experiment")
        spec = specs[0]
        spec.name = os.environ.get("EXPERIMENT_NAME", "latest_gated_history_pooled_0_24h")
        spec.description = "Latest ECG plus gated history-change; unique pooled 0-24h sequences"
        spec.target_window = "[t2-24h, t2] pooled across n=0..12"
        spec.ecg_pool, spec.use_future_query = "mean", False
        spec.use_time_embedding = True
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        spec.temporal_min_horizon_hours = None
        spec.temporal_max_horizon_hours = None
        return specs

    runner.resolve_specs = resolve
    engine.StagedModel = LatestGatedHistoryModel
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve(args.only)[0]
    data = engine.load_staged_data(spec, args)
    engine.fit(spec, args, data=data, device=device)


if __name__ == "__main__":
    main()
