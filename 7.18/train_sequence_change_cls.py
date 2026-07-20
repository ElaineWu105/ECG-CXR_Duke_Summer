#!/usr/bin/env python3
"""No-time CLS with optional learned adjacent-ECG change residuals."""
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


class SequenceChangeCLSModel(StagedModel):
    """Represent each transition using previous/current/difference/product."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.change_proj = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_model),
            nn.GELU(), nn.Linear(self.d_model, self.d_model))
        self.change_scale = nn.Parameter(torch.tensor(-2.0))

    def _encode_sequence_tokens(self, batch, add_pool_token=False):
        feats, mask = batch["ecg_feats"], batch["ecg_mask"].bool()
        if self.spec.ecg_perturb == "zero":
            return feats.new_zeros(feats.size(0), feats.size(1), self.d_model), mask, False
        batch_size = feats.size(0)
        base = self.ecg_in_proj(feats)
        change = torch.zeros_like(base)
        if base.size(1) > 1:
            previous, current = base[:, :-1], base[:, 1:]
            pair = torch.cat(
                [previous, current, current - previous, current * previous], dim=-1)
            valid = (mask[:, :-1] & mask[:, 1:]).unsqueeze(-1)
            change[:, 1:] = self.change_proj(pair) * valid
        h = base + torch.sigmoid(self.change_scale) * change
        h, mask = self._apply_sequence_token_dropout(h, mask)
        has_pool_token = False
        if add_pool_token and self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(batch_size, 1, -1), h], dim=1)
            mask = torch.cat([
                torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device), mask], dim=1)
            has_pool_token = True
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = torch.nan_to_num(self.enc_norm(h), nan=0.0, posinf=0.0, neginf=0.0)
        return h, mask, has_pool_token


def main():
    n = int(os.environ.get("N_VALUE", "2"))
    mode = os.environ.get("MODEL_MODE", "delta")
    if mode not in {"content", "delta"}:
        raise ValueError("MODEL_MODE must be content or delta")
    original_resolve = runner.resolve_specs

    def resolve(selected):
        specs = original_resolve(selected)
        if len(specs) != 1:
            raise ValueError("select one sequence experiment")
        spec = specs[0]
        spec.name = f"case2_no_time_{mode}_cls_n{n}"
        spec.description = ("No-time content+change CLS" if mode == "delta"
                            else "No-time content-only CLS")
        spec.target_window = f"[t2-{n + 12}h, t2-{n}h]"
        spec.ecg_pool, spec.use_future_query = "cls", False
        spec.use_time_embedding = False
        spec.loss_mode, spec.lambda_temporal = "cross", 0.0
        return specs

    runner.resolve_specs = resolve
    if mode == "delta":
        engine.StagedModel = SequenceChangeCLSModel
    args = runner.build_args()
    device = runner.get_device(args.device)
    spec = resolve(args.only)[0]
    data = engine.load_staged_data(spec, args)
    engine.fit(spec, args, data=data, device=device)


if __name__ == "__main__":
    main()
