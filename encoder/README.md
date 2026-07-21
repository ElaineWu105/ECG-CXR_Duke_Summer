# Frozen encoders

This directory isolates the two frozen modality encoders used to create cached features for the ECG-CXR contrastive experiments.

## BioViL-T CXR encoder

biovil_t.py loads the microsoft/BiomedVLP-BioViL-T image model from a local checkpoint. It applies the canonical grayscale chest X-ray inference transform and returns the 512-dimensional global image embedding.

## ECG-CoCa encoder

ecg_coca.py builds only the ECG tower from the ECG-CoCa configuration, loads the ecg.* checkpoint weights, and returns a normalized 512-dimensional ECG embedding for a 12-lead, 5,000-sample waveform.

Encoder weights are not included. Pass local checkpoint and model-config paths to data_preprocessing/precompute_embeddings.py.
