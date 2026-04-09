# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MRes research project adapting the **Anomaly Detection in Video Sequence with Appearance-Motion Correspondence (ICCV 2019)** architecture for **cardiac MRI anomaly detection**. The original model detected anomalies in surveillance video using a GAN that learns appearance-motion correspondence; here it's repurposed to detect cardiac pathologies from MRI images (ACDC and M&Ms datasets).

The model trains only on healthy (NOR) patients and flags anomalies as high reconstruction errors at inference time.

## Setup

```bash
pip install -r requirements.txt
```

Key dependency: `tensorflow==2.12.0`, used via TF v1 compatibility mode (`tf.compat.v1` with `tf.disable_v2_behavior()`).

## Running the Model

### Training (cardiac MRI)
```bash
python run_model.py --dataset ACDC --acdc_dir ../Dataset_2 --epochs 50
python run_model.py --dataset MM --mm_dir ../Dataset_1/Training --mm_val_dir ../Dataset_1/Validation --mm_csv ../Dataset_1/211230_M\&Ms_Dataset_information_diagnosis_opendataset.csv --epochs 50
python run_model.py --dataset COMBINED --acdc_dir ../Dataset_2 --mm_dir ../Dataset_1/Training --mm_val_dir ../Dataset_1/Validation --mm_csv ../Dataset_1/211230_M\&Ms_Dataset_information_diagnosis_opendataset.csv --epochs 100
```

RGB variant — predicts ED frame from ES frame (no optical flow; temporal difference used instead):
```bash
python run_model_rgb.py --dataset ACDC --acdc_dir ../Dataset_2 --epochs 50
```

Append extra NOR training data from reconstructed SAX volumes (both pipelines support `--recon_dir`):
```bash
python run_model.py --dataset COMBINED ... --recon_dir ../reconstructed_sax_images_training_2023
python run_model_rgb.py --dataset COMBINED ... --recon_dir ../reconstructed_sax_images_training_2023
# --recon_csv defaults to <recon_dir>/segmentation/ed_es_frames.csv
```

Resume training from a checkpoint:
```bash
python run_model.py --dataset COMBINED ... --start_epoch 30
```

### HPC Submission (Imperial College PBS)
```bash
qsub submit_job.pbs
```

### Original Video Anomaly Detection Tasks (main.py)
```bash
python main.py -d UCSDped2 -t 1        # Task 1: prepare data
python main.py -d UCSDped2 -t 2 -e 40  # Task 2: train
python main.py -d UCSDped2 -t 3 -c 0   # Task 3: test one clip
python main.py -d UCSDped2 -t 4        # Task 4: evaluate AUC
```
Available datasets: `UCSDped2`, `Avenue`, `Belleview`, `Train`, `Exit`, `Entrance`

### Jupyter Notebook
`run_model.ipynb` — interactive version of the cardiac MRI training pipeline.

## Architecture

### Core GAN (`GAN_tf.py` / `GAN_tf_rgb.py`)
Built on TF v1 graph API. Key components:
- **Generator**: U-Net with shared Inception-style encoder. Has two decoder heads — one for optical flow reconstruction and one for frame reconstruction. Skip connections are used only in the flow decoder.
- **Discriminator**: PatchGAN-style classifier taking concatenated `[frame, flow]` as input.
- **Anomaly scoring**: SSIM between real and reconstructed outputs. Lower SSIM = higher anomaly score.
- Training checkpoints saved to `training_saver/<dataset_name>/`.
- Sample generated images saved to `generated/<dataset_name>/` each epoch.
- W&B logging is integrated throughout training (`wandb.log`).

### Data Loaders
- **`data_loader.py`**: Loads ACDC (`.nii.gz`) and M&Ms cardiac MRI data. Grayscale pipeline — images are ES frames (3-channel grayscale), "flows" are Farneback optical flow between ES and ED (shape `H×W×3`: `[flow_x, flow_y, magnitude]`). Static slices filtered by `mean(magnitude) < 0.05 or max(magnitude) < 0.5`. Also contains `load_reconstructed_sax_data()` for the extra NOR `.npy` volumes.
- **`data_loader_rgb.py`**: RGB pipeline — model input is ES frame, reconstruction target is ED frame; no optical flow computed. Static slices filtered by mean pixel difference `< 0.001`. Also contains `load_reconstructed_sax_data_rgb()`. The recon `.npy` files have very small raw intensity values (`~1e-5`); the threshold is set to 0.001 (not 0.01) to avoid filtering out all recon slices.
- **`utils.py`**: Loads original video datasets (`.tif` images + precomputed `.npz` optical flow), handles ground truth labels, and computes AUC metrics.

### Input Format
All data is normalized to `[-1, 1]` range before feeding to the GAN. The model always uses 3-channel inputs (grayscale images are extended to 3 channels). Default spatial resolution: 128×128 (cardiac), 128×192 (video surveillance).

### Dataset Paths
Datasets are expected one directory level up (`../Dataset_1`, `../Dataset_2`). Video surveillance datasets go in `../dataset/`.

### TF v1 Compatibility Note
`run_model.py` and `run_model_rgb.py` mock the `ProgressBar` module and redirect `tensorflow` to `tensorflow.compat.v1`. Any new code in these scripts must remain compatible with TF v1 graph-mode semantics.
