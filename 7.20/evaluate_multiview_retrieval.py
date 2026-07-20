#!/usr/bin/env python3
"""Evaluate any same-study AP/PA/LATERAL/LL image as a correct retrieval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / "7.15"))
sys.path.insert(2, str(EXP))

from staged_dataset import StagedData, StagedDataset, collate_fn  # noqa: E402
from train_latest_gated_history import LatestGatedHistoryModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path,
                        default=ROOT / "7.20/pairs/seq_pooled_0_24h_multiview.json")
    parser.add_argument("--cxr_emb", type=Path, default=EXP / "cache/cxr_emb.npy")
    parser.add_argument("--cxr_ids", type=Path, default=EXP / "cache/cxr_ids.json")
    parser.add_argument("--ecg_emb", type=Path, default=EXP / "cache/ecg_emb.npy")
    parser.add_argument("--ecg_ids", type=Path, default=EXP / "cache/ecg_ids.json")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "7.20/multiview_retrieval_comparison.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    spec = SimpleNamespace(**checkpoint["spec"])
    config = dict(checkpoint["model_config"])
    config.pop("train_cls_only", None)
    model = LatestGatedHistoryModel(spec, **config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint["epoch"])


@torch.no_grad()
def collect_queries(model, dataset, device, batch_size):
    queries = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for batch in loader:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        query, _, _ = model.encode(moved)
        queries.append(query.float().cpu())
    return torch.cat(queries)


@torch.no_grad()
def project_gallery(model, embeddings, rows, device, batch_size):
    output = []
    for start in range(0, len(rows), batch_size):
        indices = rows[start:start + batch_size]
        values = torch.from_numpy(embeddings[indices].astype(np.float32)).to(device)
        output.append(model.cxr_proj(values).float())
    return torch.cat(output)


def summarize(ranks, selector):
    values = ranks[selector]
    return {
        "n_queries": int(values.size),
        "recall@1": float(np.mean(values <= 1)),
        "recall@5": float(np.mean(values <= 5)),
        "recall@10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_percentile_rank": None,
    }


@torch.no_grad()
def evaluate(model, dataset, positive_rows, cxr_embeddings, device, batch_size):
    query = collect_queries(model, dataset, device, batch_size)
    gallery_rows = np.unique(np.concatenate(positive_rows)).astype(np.int64)
    row_to_gallery = {int(row): i for i, row in enumerate(gallery_rows)}
    positive_gallery = [[row_to_gallery[int(row)] for row in rows] for rows in positive_rows]
    gallery = project_gallery(model, cxr_embeddings, gallery_rows, device, batch_size)
    ranks = np.empty(len(query), dtype=np.int64)

    for start in range(0, len(query), batch_size):
        end = min(start + batch_size, len(query))
        queries = query[start:end].to(device)
        similarities = queries @ gallery.t()
        positive_mask = torch.zeros_like(similarities, dtype=torch.bool)
        for local, indices in enumerate(positive_gallery[start:end]):
            positive_mask[local, indices] = True
        best_positive = similarities.masked_fill(~positive_mask, float("-inf")).max(1).values
        rank = (similarities > best_positive[:, None]).sum(1) + 1
        ranks[start:end] = rank.cpu().numpy()

    has_extra = np.asarray([len(rows) > 1 for rows in positive_rows], dtype=bool)
    all_metrics = summarize(ranks, np.ones(len(ranks), dtype=bool))
    extra_metrics = summarize(ranks, has_extra)
    for metrics in (all_metrics, extra_metrics):
        metrics["gallery_size"] = int(len(gallery_rows))
        metrics["mean_percentile_rank"] = float(
            np.mean(ranks[has_extra if metrics is extra_metrics else np.ones(len(ranks), bool)]
                    / len(gallery_rows)))
    return {"all": all_metrics, "studies_with_extra_view": extra_metrics}


def main():
    args = parse_args()
    if len(args.checkpoints) != len(args.labels):
        raise ValueError("--checkpoints and --labels must have equal length")
    device = torch.device(args.device)
    data = StagedData(
        pairs_json=str(args.pairs), kind="sequence",
        cxr_emb_npy=str(args.cxr_emb), cxr_ids_json=str(args.cxr_ids),
        ecg_emb_npy=str(args.ecg_emb), ecg_ids_json=str(args.ecg_ids), seed=args.seed,
    )
    dataset = StagedDataset(data, data.split_indices["test"], seed=args.seed + 2)
    raw = json.loads(args.pairs.read_text())["pairs"]
    cxr_index = {str(value): i for i, value in enumerate(json.loads(args.cxr_ids.read_text()))}
    positives_by_primary = {
        cxr_index[str(row["cxr_t2"])]: np.asarray(
            [cxr_index[str(value)] for value in row["cxr_positive_ids"]], dtype=np.int64)
        for row in raw if str(row["cxr_t2"]) in cxr_index
    }
    positive_rows = [positives_by_primary[int(data.pairs[index]["c2"])]
                     for index in dataset.indices]

    results = {
        "pairs": str(args.pairs), "split": "test",
        "positive_definition": "any cached AP/PA/LATERAL/LL image from the target study",
        "models": {},
    }
    for label, checkpoint_path in zip(args.labels, args.checkpoints):
        print(f"Evaluating {label}: {checkpoint_path}", flush=True)
        model, epoch = load_model(checkpoint_path, device)
        metrics = evaluate(model, dataset, positive_rows, data.cxr_emb,
                           device, args.batch_size)
        results["models"][label] = {"checkpoint": str(checkpoint_path),
                                    "checkpoint_epoch": epoch, **metrics}
        print(json.dumps(metrics, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
