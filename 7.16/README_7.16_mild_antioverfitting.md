# 7.16 milder anti-overfitting experiment

This is a separate experiment package. It does not modify the main project model script or the 7.15 model script.

Motivation: 7.15 reduced overfitting strongly, especially case2 val MRR drop, but also reduced best validation MRR compared with the 7.13 dropout rerun. 7.16 keeps the same idea but weakens the regularization.

Changes vs 7.15:

- sequence token dropout: 0.20 -> 0.10
- sequence pooled-vector dropout: 0.30 -> 0.15
- weight decay default: 3e-3 -> 1e-3
- early stopping patience default: 5 -> 7
- keeps `--sample_unique_targets`
- keeps fused embedding dropout 0.50

Run:

```bash
cd 7.16
CUDA_VISIBLE_DEVICES=1 bash run_three_cases_716.sh
```

Output:

```bash
7.16/cross_patient_huge_batch_mild_antioverfit
```
