# 7.15 anti-overfitting experiment

Goal: reduce the case2 sequence mean-pool overfitting seen in the 7.13 dropout rerun, while keeping the 21-run grid comparable to 7.13.

Methods considered:

1. Stricter early stopping: stop closer to the best validation epoch.
2. Stronger weight decay: reduce projection/head memorization.
3. Sequence token dropout: randomly hide some ECG tokens during training so case2 cannot rely on a stable patient-specific sequence signature.
4. Sequence pooled-vector LayerNorm + dropout: regularize the mean-pooled ECG sequence representation before the predictor head.
5. Unique target sampling within batch: reduce repeated-target shortcuts in batch retrieval.
6. Lower LR or smaller transformer: useful follow-ups if this run still overfits, but not enabled by default to avoid changing too many factors at once.

Implemented in this 7.15 package:

- `staged_model.py`: adds sequence token dropout p=0.20 and sequence pooled-vector dropout p=0.30. Existing fused embedding dropout p=0.50 remains.
- `run_case_experiment.py`: loads `7.15/staged_model.py` first, while keeping the rest of the package from `Waveform_CXR_EHR/ECGCXRPatientTemporal`.
- `run_three_cases_715.sh`: runs the same 3 cases x 7 n-offsets as 7.13, writes outputs to `7.15/cross_patient_huge_batch_antioverfit_seqtokdrop`, defaults to `EARLY_STOP_PATIENCE=5`, `WEIGHT_DECAY=3e-3`, and `--sample_unique_targets`.

Run:

```bash
cd ./7.15
bash run_three_cases_715.sh
```

Optional overrides:

```bash
CUDA_VISIBLE_DEVICES=3 WEIGHT_DECAY=1e-3 EARLY_STOP_PATIENCE=5 bash run_three_cases_715.sh
```
