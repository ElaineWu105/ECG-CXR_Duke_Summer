# 7.17 CLS-pooling experiment

This experiment compares the new Case 2 sequence CLS-pooling model with the
Case 3 nearest-single-ECG comparator for `n = 0, 2, 4, 6`.

- Case 2 creates a shared trainable `cls_token`, prepends it to every ECG
  sequence, lets it aggregate ECG information through Transformer
  self-attention, and uses the output at token index 0 as the sequence
  representation. No mean pooling is used in this run.
- Case 3 has one ECG rather than a sequence and therefore does not create or
  use a CLS token.
- The 7.15 model implementation and regularization are reused so that the main
  Case 2 change is mean pooling to CLS pooling.
- Both cases use cross-patient contrastive loss and the existing 7.13 pair
  files.

Run interactively:

```bash
cd ECG-CXR_Duke_Summer
bash 7.17/run_case2_cls_case3_n0_6.sh
```

Or submit to Slurm:

```bash
cd ECG-CXR_Duke_Summer
sbatch 7.17/run_case2_cls_case3_n0_6.sh
```

Outputs are written to:

```text
7.17/cross_patient_cls_pool_case2_case3/
```

To freeze the complete model and update only Case 2's `cls_token`, run:

```bash
TRAIN_CLS_ONLY=1 bash 7.17/run_case2_cls_case3_n0_6.sh
```

These outputs are kept separately in
`7.17/cross_patient_cls_only_case2_case3/`. Case 3 remains the unchanged
single-ECG comparator because it has no CLS token.

## CLS diagnostic run (n=2)

```bash
bash 7.17/run_cls_diagnostics_n2.sh
# or: sbatch 7.17/run_cls_diagnostics_n2.sh
```

This joint-training run records CLS gradient norm, parameter drift, output
diversity, and positive-negative cosine margin. The best checkpoint is also
evaluated with normal ECG, zero ECG, patient-shuffled ECG, and shuffled ECG
relative times. Outputs are under `7.17/cls_diagnostics_n2/`.
