# Data preprocessing

This directory contains the preprocessing code used before ECG-CXR contrastive training. It includes code only—no MIMIC data, patient identifiers, generated pair JSON files, embeddings, checkpoints, or server-specific paths.

## Pipeline

```text
MIMIC-CXR metadata + MIMIC-IV-ECG record list
    -> build_full_catalogs.py
    -> CXR/ECG catalog CSV files
    -> build_ecg_cxr_window_pairs.py
    -> candidate single_n*.json + seq_n*.json
    -> precompute_embeddings.py on a GPU
    -> cxr_emb.npy + cxr_ids.json + ecg_emb.npy + ecg_ids.json
    -> build_three_case_pairs.py
    -> filtered single_n*.json + seq_n*.json + nearest_n*.json
```

For offsets `n = 0,2,4,6,8,10,12`, the ECG window is `[t2-n-12h, t2-n]`, where `t2` is the target CXR time. `single` creates one pair per ECG, `seq` keeps all ECGs as one sequence, and `nearest` keeps the ECG closest to `t2` within the selected window.

## 1. Build modality catalogs

```bash
python data_preprocessing/build_full_catalogs.py \
  --cxr_metadata /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --ecg_record_list /path/to/record_list.csv \
  --ecg_root /path/to/MIMIC_IV_ECG_Matched \
  --cxr_out data_preprocessing/catalogs/cxr.csv \
  --ecg_out data_preprocessing/catalogs/ecg.csv
```

By default, this keeps subjects present in both modalities and AP/PA CXR views. It does not require an ICU stay or `hadm_id`; matching uses `subject_id` and acquisition timestamps.

## 2. Build candidate temporal pairs

```bash
python data_preprocessing/build_ecg_cxr_window_pairs.py \
  --cxr_csv data_preprocessing/catalogs/cxr.csv \
  --ecg_csv data_preprocessing/catalogs/ecg.csv \
  --metadata_path /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --cxr_root /path/to/mimic-cxr-jpg \
  --out_dir data_preprocessing/pairs
```

The generated pair JSON files include IDs, timestamps, and local path metadata so the next step can load images and waveforms. Do not commit generated pair files to a public repository.

## 3. Precompute frozen embeddings

This step requires Bio-ViL-T and ECG-CoCa checkpoints and model code. Install `torch`, `numpy`, `Pillow`, `wfdb`, `scipy`, `health-multimodal`, and the ECG-CoCa/OpenCLIP dependencies.

```bash
CUDA_VISIBLE_DEVICES=7 python data_preprocessing/precompute_embeddings.py \
  --pairs data_preprocessing/pairs/single_n{0,2,4,6,8,10,12}.json \
          data_preprocessing/pairs/seq_n{0,2,4,6,8,10,12}.json \
  --cache_dir data_preprocessing/cache \
  --biovil_ckpt /path/to/biovil_t_image_model_proj_size_128.pt \
  --ecg_ckpt /path/to/cpt_wfep_epoch_20.pt \
  --ecg_config /path/to/coca_ViT-B-32.json \
  --device cuda \
  --num_workers 8
```

Rows and IDs have the same ordering:

```text
cxr_emb.npy[i] <-> cxr_ids.json[i]
ecg_emb.npy[i] <-> ecg_ids.json[i]
```

Use `--merge` to preserve an existing cache and encode only new IDs. Use `--max_items 100` for a smoke test before a full run.

## 4. Build the final three cases

After generating the embedding ID files, run the exact 7.13 case builder:

```bash
python data_preprocessing/build_three_case_pairs.py \
  --cxr_times /path/to/cxr_study_times.csv \
  --ecg_times /path/to/ecg_record_times.csv \
  --cxr_metadata /path/to/mimic-cxr-metadata.csv \
  --cxr_ids data_preprocessing/cache/cxr_ids.json \
  --ecg_ids data_preprocessing/cache/ecg_ids.json \
  --output_dir data_preprocessing/pairs/final
```

This filters to embedded AP/PA CXR images and ECG records, then writes the `single`, `seq`, and `nearest` pair families for n=0–12.
