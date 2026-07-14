# Deepfake Detection with Xception-based Architectures — Final Project, Artificial Vision (I308)

This repository contains the code and notebooks for the Final Project of Artificial Vision (I308),
on deepfake detection and cross-generalization using FaceForensics++.

> **Note:** the `frames/` folder (the extracted and preprocessed dataset, ~2GB) is not included in
> this repository due to its size. To reproduce the work from scratch, follow the steps in this
> README to download the dataset and regenerate the frames locally.

## 1. Requirements

```bash
pip install -r requirements.txt
```

This project requires Python 3.10+ and (recommended) a CUDA-capable GPU for model training
(Sections 3 onward). Preprocessing (step 2) can be run on CPU.

## 2. Dataset download (FaceForensics++)

1. Download the videos with `c23` compression (constant quality, CRF 23).
2. Save the downloaded videos following the folder structure provided by the official download script.

`DeepFakeDetection` **does not need to be downloaded** to reproduce this work: it was excluded from
all experiments due to a naming-scheme incompatibility with the official splits (see Section III of
the report).

## 3. Frame extraction

Once the videos are downloaded, run the extraction script:

```bash
python 00_extract_frames.py --input_dir <path_to_downloaded_videos> --output_dir frames/
```

This extracts 10 uniformly distributed frames per video, resizes them to 299×299 pixels, and saves
them in `frames/`, organized by class (real + 5 manipulation methods).

## 4. Train/validation/test splits

```bash
jupyter notebook 02_prepare_splits.ipynb
```

This notebook builds the official FaceForensics++ splits (defined at the video-pair level, not the
individual frame level, to avoid identity-based data leakage) and saves the result to `splits/` and
`dataset_split/`.

## 5. Notebook execution order

The notebooks are numbered in the order they should be run. Each one assumes the previous ones have
already been executed and that their outputs (checkpoints, CSVs) are available in `outputs/`.

- `00_extract_frames.py`: Extracts frames from the downloaded videos
- `01_exploratory_analysis.ipynb`: Exploratory analysis of the dataset (class distribution, image quality)
- `02_prepare_splits.ipynb`: Builds official train/val/test splits
- `03a_train_xception.ipynb`: Trains the baseline Xception model
- `03b_train_efficientnet.ipynb`: Trains the baseline EfficientNet-B4 model
- `04_train_single_method.ipynb`: Trains 5 Xception instances, each specialized in a single manipulation method (cross-generalization protocol)
- `05_cross_generalization_matrix.ipynb`: Evaluates the 5 specialized models against each other, generating the 5×5 AUC matrix
- `06_xception_FNO.ipynb`: Trains and evaluates XceptionFNO (Fourier Neural Operator branch)
- `07_XceptionBiFPN.ipynb`: Trains and evaluates XceptionBiFPN (multi-scale fusion)
- `08_grad_cam.ipynb`: Generates Grad-CAM visualizations for the three architectures
- `09_umbral.ipynb`: Calibrates the decision threshold via Youden's index, computes metrics (F1, FNR, FPR, Precision)
- `10_error_analysis.ipynb`: Error analysis: FNR by method, failure consensus across models

## 6. Expected folder structure

```
.
├── frames/              # generated in step 3
├── dataset_split/        # generated in step 4
├── splits/                # generated in step 4
├── outputs/              # checkpoints (.pth), result CSVs, generated figures
├── src/                   # shared modules (config, data, models, engine)
├── 00_extract_frames.py
├── 01_exploratory_analysis.ipynb
├── ...
```
