#!/usr/bin/env python3
"""No-time CLS with an explicit whole-window ECG change token."""
from __future__ import annotations
import os, sys
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


class WindowChangeCLSModel(StagedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d_model
        self.token_change_proj = nn.Sequential(
            nn.Linear(3*d, d, bias=False), nn.GELU(), nn.Linear(d, d, bias=False))
        self.token_change_scale = nn.Parameter(torch.tensor(-2.0))
        self.local_change_score = nn.Linear(d, 1)
        self.window_change_proj = nn.Sequential(
            nn.Linear(4*d, d), nn.GELU(), nn.Linear(d, d))
        self.window_change_scale = nn.Parameter(torch.tensor(-2.0))

    def _changes(self, base, mask):
        local = torch.zeros_like(base)
        if base.size(1) > 1:
            valid = mask[:, 1:] & mask[:, :-1]
            local[:, 1:] = (base[:, 1:] - base[:, :-1]) * valid.unsqueeze(-1)
        cumulative = (base - base[:, :1]) * mask.unsqueeze(-1)
        return local, cumulative

    def _window_token(self, base, local, mask):
        batch_size = base.size(0)
        lengths = mask.sum(dim=1).long()
        multi = lengths > 1
        last = base[torch.arange(batch_size, device=base.device),
                    (lengths - 1).clamp(min=0)]
        first = base[:, 0]
        local_valid = mask.clone()
        local_valid[:, 0] = False
        scores = self.local_change_score(local).squeeze(-1).masked_fill(~local_valid, -1e9)
        weights = torch.softmax(scores, dim=1) * local_valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        local_pool = (local * weights.unsqueeze(-1)).sum(dim=1)
        summary = torch.cat([first, last, last-first, local_pool], dim=-1)
        token = torch.sigmoid(self.window_change_scale) * self.window_change_proj(summary)
        return token * multi.unsqueeze(-1), multi

    def _encode_sequence_tokens(self, batch, add_pool_token=False):
        feats, mask = batch["ecg_feats"], batch["ecg_mask"].bool()
        if self.spec.ecg_perturb == "zero":
            return feats.new_zeros(feats.size(0), feats.size(1), self.d_model), mask, False
        batch_size = feats.size(0)
        base = self.ecg_in_proj(feats)
        local, cumulative = self._changes(base, mask)
        change_input = torch.cat([local, cumulative, local*cumulative], dim=-1)
        h = base + torch.sigmoid(self.token_change_scale) * self.token_change_proj(change_input)
        window_token, window_valid = self._window_token(base, local, mask)
        h, mask = self._apply_sequence_token_dropout(h, mask)
        has_pool_token = False
        if add_pool_token and self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, 1, -1)
            h = torch.cat([cls, window_token.unsqueeze(1), h], dim=1)
            cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_valid, window_valid.unsqueeze(1), mask], dim=1)
            has_pool_token = True
        h = self.encoder(h, src_key_padding_mask=~mask)
        return torch.nan_to_num(self.enc_norm(h), nan=0.0, posinf=0.0, neginf=0.0), mask, has_pool_token


def main():
    n = int(os.environ.get("N_VALUE", "2"))
    original_resolve = runner.resolve_specs
    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("select exactly one sequence experiment")
        spec = specs[0]
        spec.name = f"case2_no_time_window_change_cls_n{n}"
        spec.description = "No-time CLS with local, cumulative, and global ECG change"
        spec.target_window = f"[t2-{n+12}h, t2-{n}h]"
        spec.ecg_pool, spec.use_future_query = "cls", False
        spec.use_time_embedding = False
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        return specs
    runner.resolve_specs = resolve
    engine.StagedModel = WindowChangeCLSModel
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve(args.only)[0]
    data = engine.load_staged_data(spec, args)
    engine.fit(spec, args, data=data, device=device)


if __name__ == "__main__":
    main()
