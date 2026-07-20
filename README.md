# ECG-to-CXR Contrastive Learning: July 18-20

This snapshot contains the main model and objective code for three days of ECG-to-CXR retrieval experiments using frozen ECG and CXR embeddings. Patient splits are disjoint.

No MIMIC data, embeddings, pair files, checkpoints, logs, or patient identifiers are included. MIMIC access and derived data remain subject to the PhysioNet data-use agreement.

## July 18: sequence and change diagnostics

We tested whether multiple ECGs add temporal information beyond the nearest ECG. Experiments compared sequence CLS pooling with nearest-ECG retrieval, added within-patient temporal InfoNCE, and evaluated content-only, adjacent-change, and whole-window-change sequence encoders at n=0,2,4.

Temporal supervision improved within-patient temporal ranking but reduced cross-patient retrieval. Explicit change encoders were inconsistent. Paired analysis showed that most rows contain one ECG, leaving the multi-ECG subset too small for stable gains.

Main files: `7.18/train_sequence_change_cls.py` and `7.18/train_window_change_cls.py`.

## July 19: label-guided multi-positive learning

The instance objective was extended to:

```text
L = L_instance_InfoNCE + lambda_label * L_label_multi_positive
```

Different-patient CXRs become soft positives when they share explicitly positive CheXpert labels. Positive weights use label-set Jaccard similarity; uncertain and missing labels are masked. The controlled n=2 experiment compared all positives, top-10 positives, fixed lambda, and learnable lambda.

The learnable top-10 run reached R@1 0.7639%, R@5 2.9829%, R@10 4.7472%, and MRR 2.4677%, but did not beat the original baseline on every metric.

Main files: `7.19/train_label_multipositive.py` and `7.19/train_label_multipositive_learnable.py`.

## July 20: pooled history, multiview positives, and prototypes

We pooled the seven overlapping n=0,2,...,12 windows into one unique [t2-24h,t2] row per target CXR and removed duplicate ECGs. A gated-history model uses the latest ECG as current state and older ECGs as an attention-pooled residual change.

Same-study AP/PA/LATERAL/LL images were added as positives, followed by six CheXpert disease prototypes. The final controlled loss was:

```text
L = L_primary_InfoNCE + 0.02 * L_multiview + 0.1 * L_prototype
```

We ran 3 ECG cases (all ECGs as separate pairs, sequence, nearest ECG) across n=0,2,4,6,8,10,12, for 21 experiments, plus a prototype ablation. Sequence modeling was not consistently better than the simpler all-ECG or nearest-ECG cases. The prototype ablation was nearly neutral.

Main files: `7.20/train_latest_gated_history.py`, `7.20/train_latest_gated_history_multiview.py`, and `7.20/train_primary_multiview_prototype.py`.

## Layout

```text
7.15/staged_model.py                 shared model
7.18/                                sequence/change experiments
7.19/                                label multi-positive objectives
7.20/                                gated history and multiview/prototype runs
Waveform_CXR_EHR/ECGCXRPatientTemporal/
                                      engine, losses, data, sampler, metrics
results/                              aggregate de-identified metrics
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required local inputs

Place frozen embeddings under:

```text
Waveform_CXR_EHR/ECGCXRPatientTemporal/cache/
  ecg_emb.npy  ecg_ids.json
  cxr_emb.npy  cxr_ids.json
```

Pair JSON files are expected under `7.13/pairs/`; pooled files are written under `7.20/pairs/`. They are intentionally excluded. Configure MIMIC-CXR labels with:

```bash
export MIMIC_CXR_ROOT=/path/to/mimic-cxr-jpg
export LABELS_CSV="$MIMIC_CXR_ROOT/mimic-cxr-2.0.0-chexpert.csv.gz"
```

## Main runs

```bash
# Pooled 24-hour gated-history baseline
CUDA_VISIBLE_DEVICES=0 bash 7.20/run_latest_gated_history_natural.sh

# Primary plus multiview plus disease prototypes
CUDA_VISIBLE_DEVICES=0 LAMBDA_MULTIVIEW=0.02 LAMBDA_PROTOTYPE=0.1 \
  bash 7.20/run_primary_multiview_prototype.sh

# 21-run grid followed by best-case prototype ablation
CUDA_VISIBLE_DEVICES=0 bash 7.20/run_grid_then_ablation_gpu7.sh
```

The grid skips experiments that already contain `results.json`.

## Results

See `results/three_cases_summary.csv`. Because query count and gallery size change across n, compare the three cases primarily within the same n.

Best observed test values, not all from the same configuration:

- R@1 0.8818%: nearest ECG, n=6
- R@5 3.0377%: nearest ECG, n=4
- R@10 5.1676%: nearest ECG, n=4
- MRR 2.5880%: all ECGs, n=6

These are single-seed experimental results, not confidence-bounded improvements.
