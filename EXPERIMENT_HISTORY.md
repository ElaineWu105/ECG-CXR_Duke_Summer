# Experiment history: June 30 to July 20

Only source code, shell entry points, figures, and aggregate de-identified results are included. Raw MIMIC data, patient-level tables, pair JSON files, embedding caches, checkpoints, logs, and patient splits are excluded.

- **June 30:** compared ECG and CXR patient coverage and analyzed CXR intervals.
- **July 1:** parsed ECG/CXR timestamps, counted temporal-window pairs for n=0-12, and plotted pair-count and distribution diagnostics.
- **July 6:** debugged pair statistics and summarized CXR-count distributions.
- **July 7:** filtered frontal CXR views and prototyped Criterion-2 sequence pair construction.
- **July 13:** built the single, sequence, and nearest pair families; ran the 21 cross-patient contrastive experiments; reran with embedding dropout; and generated batch-level train/validation diagnostics.
- **July 14:** evaluated six-label downstream classifiers and compared uncertain/missing-label policies across the three ECG representations and n=0-12.
- **July 15:** applied stronger anti-overfitting controls to the three-case contrastive grid.
- **July 16:** relaxed regularization and reran the full 21-experiment grid.
- **July 17:** tested CLS pooling, time gating, and input/gradient diagnostics.
- **July 18:** tested sequence-change encoders and temporal supervision.
- **July 19:** added label-guided multi-positive contrastive objectives with fixed and learnable loss weights.
- **July 20:** tested pooled history, same-study multiview positives, lateral views, and disease prototypes.

See each dated directory for runnable scripts and aggregate outputs. See data_preprocessing/ for catalog, pair, and embedding-cache construction, and encoder/ for the isolated BioViL-T and ECG-CoCa wrappers.
