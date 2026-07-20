#!/usr/bin/env python3
"""Time ablations on multi-ECG queries against the full split CXR gallery."""
from __future__ import annotations

import argparse
import csv
import json
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

import config as C  # noqa: E402
import engine  # noqa: E402
import metrics  # noqa: E402
import run_experiments as runner  # noqa: E402
from staged_dataset import StagedDataset, collate_fn  # noqa: E402
from train_cls_diagnostics import DiagnosticStagedModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(
        HERE / "cls_diagnostics_n2/case2_cls_diagnostics_n2/best.pt"))
    p.add_argument("--output_dir", default=str(
        HERE / "cls_diagnostics_n2/case2_cls_diagnostics_n2/multi_ecg_time_ablation"))
    p.add_argument("--pairs", default=str(ROOT / "7.13/pairs/seq/seq_n2.json"))
    p.add_argument("--cxr_emb", default=str(EXP / "cache/cxr_emb.npy"))
    p.add_argument("--cxr_ids", default=str(EXP / "cache/cxr_ids.json"))
    p.add_argument("--ecg_emb", default=str(EXP / "cache/ecg_emb.npy"))
    p.add_argument("--ecg_ids", default=str(EXP / "cache/ecg_ids.json"))
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def make_spec():
    spec = runner.resolve_specs(["exp3a_seq_ecg_meanpool"])[0]
    spec.name = "case2_cls_diagnostics_n2"
    spec.target_window = "[t2-14h, t2-2h]"
    spec.ecg_pool = "cls"
    spec.use_future_query = False
    spec.loss_mode = "cross"
    spec.lambda_temporal = 0.0
    return spec


def load_model(spec, data, checkpoint, device):
    saved = torch.load(checkpoint, map_location=device)
    cfg = saved["model_config"]
    model = DiagnosticStagedModel(
        spec, cxr_dim=data.cxr_emb.shape[1], ecg_dim=data.ecg_emb.shape[1],
        proj_dim=cfg["proj_dim"], cxr_proj_hidden=cfg["cxr_proj_hidden"],
        d_model=cfg["d_model"], ecg_tx_layers=cfg["ecg_tx_layers"],
        ecg_tx_heads=cfg["ecg_tx_heads"], ecg_tx_mlp_ratio=cfg["ecg_tx_mlp_ratio"],
        fusion_hidden=cfg["fusion_hidden"], time_emb_dim=cfg["time_emb_dim"],
        dropout=cfg["dropout"], temperature=cfg["temperature"],
        learnable_temperature=cfg["learnable_temperature"],
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def perturb_time(batch, mode):
    if mode == "normal":
        return batch
    b = dict(batch)
    times = batch["ecg_t2t"].clone()
    for i, length in enumerate(batch["ecg_mask"].sum(dim=1).tolist()):
        length = int(length)
        if mode == "reversed_time":
            times[i, :length] = times[i, :length].flip(0)
        elif mode == "permuted_time":
            times[i, :length] = times[i, :length].roll(1)
        elif mode == "constant_time":
            times[i, :length] = 8.0
        else:
            raise ValueError(mode)
    b["ecg_t2t"] = times
    return b


@torch.no_grad()
def collect_queries(model, dataset, device, batch_size, mode):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    queries, targets = [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q, _, _ = model.encode(perturb_time(b, mode))
        queries.append(q.float().cpu())
        targets.append(batch["c2_row"])
    return torch.cat(queries), torch.cat(targets).numpy()


def cross_metrics(q, target_rows, gallery, gallery_rows):
    row_to_col = {int(row): i for i, row in enumerate(gallery_rows)}
    targets = np.asarray([row_to_col[int(row)] for row in target_rows])
    sims = (q @ gallery.t()).numpy()
    ranks = []
    for i, target in enumerate(targets):
        ranks.append(metrics._rank_of_target(sims[i], int(target)))
    ranks = np.asarray(ranks)
    n = len(ranks)
    return {
        "n_queries": n,
        "gallery_size": len(gallery_rows),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def main():
    args = parse_args()
    device = runner.get_device(args.device)
    spec = make_spec()
    data_args = argparse.Namespace(
        seq_target_pairs=args.pairs, single_pairs="", pairs="",
        cxr_emb=args.cxr_emb, cxr_ids=args.cxr_ids,
        ecg_emb=args.ecg_emb, ecg_ids=args.ecg_ids, seed=args.seed)
    data = engine.load_staged_data(spec, data_args)
    model = load_model(spec, data, args.checkpoint, device)
    output = {}
    modes = ["normal", "reversed_time", "permuted_time", "constant_time"]

    for split in ("val", "test"):
        full_indices = data.split_indices[split]
        multi_indices = [i for i in full_indices if len(data.pairs[i]["ecg_rows"]) > 1]
        query_ds = StagedDataset(data, multi_indices, ecg_perturb="none", seed=args.seed)
        gallery_rows = np.unique([data.pairs[i]["c2"] for i in full_indices])
        gallery = metrics._gallery(model, data.cxr_emb, gallery_rows, device, args.batch_size)
        output[split] = {}
        for mode in modes:
            q, targets = collect_queries(model, query_ds, device, args.batch_size, mode)
            output[split][mode] = cross_metrics(q, targets, gallery, gallery_rows)
            print(split, mode, output[split][mode])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, indent=2))
    rows = []
    for split, variants in output.items():
        for mode, values in variants.items():
            rows.append({"split": split, "variant": mode, **values})
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out_dir / "results.csv")


if __name__ == "__main__":
    main()
