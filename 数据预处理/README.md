# 数据预处理

本目录包含 ECG–CXR contrastive learning 训练前的完整数据处理代码。它只提供代码，不包含 MIMIC 原始数据、患者标识、pair JSON、embedding、模型权重或服务器路径。

## 处理流程

```text
MIMIC-CXR metadata + MIMIC-IV-ECG record list
    -> build_full_catalogs.py
    -> CXR/ECG catalog CSV
    -> build_ecg_cxr_window_pairs.py
    -> single_n*.json + seq_n*.json
    -> candidate single_n*.json + seq_n*.json
    -> precompute_embeddings.py (GPU)
    -> cxr_emb.npy + cxr_ids.json + ecg_emb.npy + ecg_ids.json
    -> build_three_case_pairs.py
    -> filtered single_n*.json + seq_n*.json + nearest_n*.json
```

`n = 0,2,4,6,8,10,12`，ECG 窗口为 `[t2-n-12h, t2-n]`，其中 `t2` 是目标 CXR 时间。`single` 将窗口内每条 ECG 分别配到 CXR，`seq` 将全部 ECG 组成一个序列，`nearest` 只保留窗口内离 `t2` 最近的 ECG。

## 1. 建立 catalog

```bash
python 数据预处理/build_full_catalogs.py \
  --cxr_metadata /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --ecg_record_list /path/to/record_list.csv \
  --ecg_root /path/to/MIMIC_IV_ECG_Matched \
  --cxr_out 数据预处理/catalogs/cxr.csv \
  --ecg_out 数据预处理/catalogs/ecg.csv
```

默认只保留同时出现在 ECG 与 CXR 中的患者，并保留 AP/PA CXR。这里不要求 ICU stay 或 `hadm_id`；匹配依据是同一 `subject_id` 和采集时间。

## 2. 建立 pair

```bash
python 数据预处理/build_ecg_cxr_window_pairs.py \
  --cxr_csv 数据预处理/catalogs/cxr.csv \
  --ecg_csv 数据预处理/catalogs/ecg.csv \
  --metadata_path /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --cxr_root /path/to/mimic-cxr-jpg \
  --out_dir 数据预处理/pairs

```

pair JSON 保存 ID、时间和本地文件路径 metadata，供下一步读取原始图像与波形；因此不要把生成的 JSON 提交到公开仓库。

## 3. 生成 embedding cache

需要 Bio-ViL-T checkpoint、ECG-CoCa checkpoint、ECG-CoCa model config，并安装 `torch`, `numpy`, `Pillow`, `wfdb`, `scipy`, `health-multimodal` 以及 ECG-CoCa/OpenCLIP 代码。

```bash
CUDA_VISIBLE_DEVICES=7 python 数据预处理/precompute_embeddings.py \
  --pairs 数据预处理/pairs/single_n{0,2,4,6,8,10,12}.json \
          数据预处理/pairs/seq_n{0,2,4,6,8,10,12}.json \
  --cache_dir 数据预处理/cache \
  --biovil_ckpt /path/to/biovil_t_image_model_proj_size_128.pt \
  --ecg_ckpt /path/to/cpt_wfep_epoch_20.pt \
  --ecg_config /path/to/coca_ViT-B-32.json \
  --device cuda \
  --num_workers 8
```

输出文件的第 `i` 行和 ID JSON 的第 `i` 项严格对应：

```text
cxr_emb.npy[i] <-> cxr_ids.json[i]
ecg_emb.npy[i] <-> ecg_ids.json[i]
```

增加新的 pair 文件时可以加入 `--merge`，只计算缓存中尚不存在的 ID。正式运行前可加入 `--max_items 100` 做 smoke test。

## 4. 生成最终三种 case

已有 embedding ID 后，运行 7.13 实际使用的过滤脚本：

```bash
python 数据预处理/build_three_case_pairs.py \
  --cxr_times /path/to/cxr_study_times.csv \
  --ecg_times /path/to/ecg_record_times.csv \
  --cxr_metadata /path/to/mimic-cxr-metadata.csv \
  --cxr_ids 数据预处理/cache/cxr_ids.json \
  --ecg_ids 数据预处理/cache/ecg_ids.json \
  --output_dir 数据预处理/pairs/final
```

这一步只保留已有 embedding 的 AP/PA 图像和 ECG，并输出 `single`、`seq`、`nearest` 三组 n=0–12 pair。
