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

`run_model.py` is the unified entry point for cardiac MRI training. Three orthogonal flags control every run:

| Flag | Options | Effect |
|---|---|---|
| `--model_type` | `flow` (default) / `rgb` | Which GAN backend: `GAN_tf` (optical flow head) or `GAN_tf_rgb` (ED-frame prediction head) |
| `--datasets` | `ACDC` `MM` `RECON` (one or more, space-separated) | Training datasets to load; `RECON` requires `--recon_dir` |
| `--frame_mode` | `ed_es` (default) / `next_frame` / `next_frame_systole` | ED/ES pairs only · all consecutive frame pairs · consecutive pairs restricted to t ∈ [ED, ES−1] (systolic phase only; patients with ES ≤ ED are skipped). For RECON, the ED/ES CSV (`--recon_csv` or `<recon_dir>/segmentation/ed_es_frames.csv`) is required when using `next_frame_systole`; cases missing from the CSV are skipped. Validation also restricts to the same systolic range. |

Optional orientation-normalisation flags (default off, both loaders pass-through when disabled):

| Flag | Default | Effect |
|---|---|---|
| `--orient_normalize` | off | Rotate + translate every 4D volume so the LV centroid sits at the image centre and the RV pool sits on the viewer's left. Applied per patient (single transform per case) using params keyed by `case_id`. |
| `--orient_params` | `<recon_dir>/segmentation/orientation_params.csv` (or workspace-relative fallback) | Path to the precomputed orientation CSV produced by `reconstructed_sax_images_training_2023/compute_orientation.py`. |

Optional spacing-normalisation flags (default off, both loaders pass-through when disabled). When on, applied AFTER orientation:

| Flag | Default | Effect |
|---|---|---|
| `--spacing_normalize` | off | Resample each volume to `--target_spacing` mm/px (in-plane) then centre-crop or zero-pad to `--target_size` px. Default 1.5 mm/px × 128 px = 192 mm × 192 mm FoV. Output array is `(T, Z, target_size, target_size)`. |
| `--target_spacing` | `1.5` (mm/px) | Target in-plane voxel spacing after resample. |
| `--target_size` | `128` (px) | Output side length. With default spacing this is a 192 mm box. |
| `--recon_spacing` | `2.0` (mm/px) | Assumed in-plane spacing for RECON `.npy` volumes (no NIfTI header on disk). Cleanest place to override if upstream reconstruction changes. |

```bash
# Flow model, ACDC + MM, ED/ES pairs (the main cardiac MRI experiment)
python run_model.py --model_type flow --datasets ACDC MM --frame_mode ed_es --epochs 100 \
    --acdc_dir ../Dataset_2 \
    --mm_dir ../Dataset_1/Training \
    --mm_val_dir ../Dataset_1/Validation \
    --mm_csv "../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

# RGB model, all three datasets
python run_model.py --model_type rgb --datasets ACDC MM RECON --frame_mode ed_es \
    --recon_dir ../reconstructed_sax_images_training_2023 --epochs 100 ...

# Next-frame training (uses all consecutive pairs, not just ED/ES)
python run_model.py --model_type flow --datasets ACDC MM --frame_mode next_frame --epochs 100 ...

# Resume from checkpoint epoch N
python run_model.py ... --start_epoch 30 --epochs 100
```

The checkpoint folder name encodes all three choices, e.g. `ACDC_MM_ED_ES_FLOW_NOR`. `run_model_rgb.py` also exists as a standalone RGB-only script with the same `--datasets` / `--frame_mode` flags.

### HPC Submission (Imperial College PBS)
```bash
qsub submit_job.pbs
```
Note: `submit_job.pbs` still uses the old `--dataset COMBINED` syntax — update it to use the new flags before submitting.

### Original Video Anomaly Detection Tasks (main.py)
```bash
python main.py -d UCSDped2 -t 1        # Task 1: prepare data
python main.py -d UCSDped2 -t 2 -e 40  # Task 2: train
python main.py -d UCSDped2 -t 3 -c 0   # Task 3: test one clip
python main.py -d UCSDped2 -t 4        # evaluate AUC
```
Available datasets: `UCSDped2`, `Avenue`, `Belleview`, `Train`, `Exit`, `Entrance`

### Jupyter Notebook
`run_model.ipynb` — interactive version of the cardiac MRI training pipeline. May use older API.

## Architecture

### Core GAN (`GAN_tf.py` / `GAN_tf_rgb.py`)

Both files share the same layer primitives (`conv2d`, `conv_transpose`, `conv2d_Inception`) and overall U-Net structure, built on the TF v1 graph API.

**Generator** — shared Inception-style encoder (`h0`–`h5`), then two independent decoder heads:
- **Auxiliary decoder** (with skip connections from `h1`–`h4`): predicts optical flow (`GAN_tf`) or the ED frame from ES (`GAN_tf_rgb`). Final output bounded by `tanh`.
- **Reconstruction decoder** (no skip connections): always reconstructs the input (ES) frame. Final output bounded by `tanh`. No skip connections force a full bottleneck pass, making reconstruction harder for out-of-distribution (anomalous) inputs.
- Decoder blocks follow `deconv → BN → leaky_relu → dropout → concat_skip` ordering throughout.

**Discriminator** — PatchGAN-style; takes `concat([frame_true, flow_hat], axis=-1)` as input.

**Loss** — `G_loss_total = lw_adv×G_adv + lw_appe×loss_appe + lw_aux×loss_aux`
- `loss_appe`: MSE + gradient loss between reconstructed and input frame.
- `loss_aux`: L1 loss between predicted flow and GT flow (`GAN_tf`); MSE + gradient loss between predicted ED and GT ED frame (`GAN_tf_rgb`).
- Default weights: `lw_adv=0.25, lw_appe=1.0, lw_aux=2.0` (passed as kwargs to `train_Unet_naive_with_batch_norm`; exposed as `--lw_adv / --lw_appe / --lw_aux` in `run_model.py`).

**Anomaly scoring at validation** — three parallel scoring pipelines run every validation epoch:

**Full-frame:** per-sample `loss_appe` and `loss_aux` reduced over all spatial dims. Combined score:
```
combined = log(loss_aux / μ_aux) + 0.2 × log(loss_appe / μ_appe)
```
Logged to W&B as `Val_AUC_Appearance`, `Val_AUC_Flow` / `Val_AUC_ED`, `Val_AUC_Combined`.

**Patch-based (paper Section 3.5):** raw 2D MSE diff maps `[B, H, W]` computed by reducing over channels only. `compute_patch_scores()` in `utils.py` wraps `find_max_patch()` (16×16 patch, stride 4) to select the patch with highest mean auxiliary MSE, then reads appearance MSE at that same position. Combined score:
```
combined_patch = log(S_F(P̃) / μ_F) + 0.2 × log(S_I(P̃) / μ_I)
```
Logged to W&B as `Val_AUC_Appe_Patch`, `Val_AUC_Flow_Patch` / `Val_AUC_ED_Patch`, `Val_AUC_Combined_Patch`. Printed as `[VAL-AUC-PATCH]`.

**Per-disease one-vs-NOR AUC:** for each disease label (MINF, DCM, HCM, RV), computes AUC treating that disease vs NOR only. Logged as `Val_AUC_<DISEASE>_Appearance` and `Val_AUC_<DISEASE>_Flow` / `_ED`. Printed as `[VAL-AUC-<DISEASE>]`.

**Optimal weight search:** sweeps the appearance weight `w` in `log(aux/μ_aux) + w·log(appe/μ_appe)` over 21 points in [0, 1] using the current val set each epoch. Logs `Val_AUC_BestCombined` and `Val_BestWeight_Appe` — a ceiling reference for what the combined score can achieve with an optimally tuned weight. Printed as `[VAL-OPT-WEIGHT]`.

In all pipelines, `μ_*` are the mean per-sample scores computed over the **entire training set in eval mode**, recomputed each validation epoch. Training baselines logged as `Train_Baseline_Appe` / `Train_Baseline_Opt` (`GAN_tf`) or `Train_Baseline_ED` (`GAN_tf_rgb`). Note: the training-set eval pass runs every validation epoch — roughly doubles validation wall time.

**`GAN_tf_rgb.py` differences from `GAN_tf.py`:**
- `augment_paired_batch()` is defined but no longer called from the training loop. It applied a random 90/180/270° rotation to (ES, ED) pairs, which now defeats orientation normalisation — once the loader puts every patient in a canonical pose (LV centred, RV on viewer left), randomising the orientation back via 90° increments undoes that canonicalisation. Pass-through line at the call site preserves the variable names so the rest of the loop is unchanged. Re-enable by uncommenting the one line marked in `GAN_tf_rgb.py`.
- Auxiliary target is `scaled_ed` (ED frame in `[-1, 1]`); the discriminator sees `[es_frame, pred_ed]`.
- `loss_aux` is named `loss_ed`; W&B keys use `_ED` suffix instead of `_Flow`.

**Data augmentation across the codebase**: the 90° rotation in `augment_paired_batch` is the *only* data augmentation present anywhere — there's nothing in `GAN_tf.py` (flow head), in either data loader, or in `utils.py`. With that disabled, training runs on the canonical post-normalisation frames with no augmentation. Future augmentations should be implemented in the loaders or inside the training loop alongside the disabled `augment_paired_batch` call.

### Data Loaders

`run_model.py` imports both loaders as `dl_flow` / `dl_rgb`; they are never mixed within a single run.

**`data_loader.py`** — grayscale pipeline, returns `(images, flows)`:

| Function | Description |
|---|---|
| `load_acdc_data` | ACDC NOR training, all consecutive frame pairs |
| `load_mm_data` | M&M NOR training, all consecutive frame pairs |
| `load_combined_data` | ACDC + M&M, consecutive pairs |
| `load_acdc_ed_es_data` | ACDC NOR training, ED/ES pairs only |
| `load_mm_ed_es_data` | M&M NOR training, ED/ES pairs only |
| `load_combined_ed_es_data` | ACDC + M&M combined, ED/ES pairs only |
| `load_reconstructed_sax_data` | RECON `.npy` volumes, ED/ES pairs |
| `load_reconstructed_sax_data_next_frame` | RECON `.npy` volumes, consecutive pairs |
| `load_acdc_test_val_ed_es_data` | ACDC test set split into val/test, ED/ES only (fixed seed=42) |
| `load_mm_validation_ed_es_data` | M&M Validation folder, all pathologies, ED/ES only |

**`data_loader_rgb.py`** — RGB pipeline, returns `(es_images, ed_images)`. Mirrors the above with the same function names plus `load_reconstructed_sax_data_rgb` / `load_reconstructed_sax_data_next_frame_rgb`. Core helpers: `extract_consecutive_pairs` and `extract_edes_pairs` do the actual frame extraction; the dataset-level functions call these.

**Preprocessing (both pipelines):**
- Frames normalised to `[0, 1]` via 1st/99th percentile clipping, then converted to 3-channel.
- Resized with `aspect_preserve_resize` (letterbox to 128×128).
- Static slices discarded: `mean(flow_mag) < 0.05` or `max(flow_mag) < 0.5` (flow pipeline); `mean(|f2 - f1|) < 0.01` (RGB pipeline); threshold is `0.001` for RECON volumes (raw intensities ~1e-5).
- **Slice selection**: every per-slice loop iterates `_middle_slice_range(Z)`, which drops the top and bottom 20% of slices (keeps the middle 60%). Helper defined at the top of each loader file. For small Z (e.g. Z ≤ 5) it falls back to dropping a single slice from each end so the previous `range(1, Z - 1)` behaviour is preserved. Z=8 ends up at 50% because round(1.6)=2; Z=10/15/20 land at 60% cleanly.
- The GAN receives frames in `[-1, 1]` — scaling is applied inside the TF graph (`(x / 0.5) − 1`), not in the loaders.

**Orientation normalisation (opt-in via `--orient_normalize`):**
- Each loader exposes `set_orientation_normalization(enabled, csv_path)` which `run_model.py` calls before any `load_*` invocation. When enabled, every 4D volume returned by the load step is rotated + translated in the *original* image coordinate frame before any percentile normalisation / resize. Computed once per patient (single transform applied to every (T, Z) slice) so temporal coherence is preserved.
- Implementation: `orientation_normalize.py` (loader side, dependency-light — `cv2.warpAffine` only) reads the per-patient row from `orientation_params.csv` (`case_id, dataset, lv_cy, lv_cx, rv_cy, rv_cx, delta_deg, tx, ty, flip, status, ...`) and applies the affine. In `data_loader.py` the helper `_maybe_normalize_volume(volume, case_id)` is called explicitly at every load site; in `data_loader_rgb.py` the wrap is inside `load_and_orient_sitk()`, which derives `case_id` from the filename (skipping `_gt.nii.gz` masks). The two RECON `np.load(...)` sites in `data_loader_rgb.py` are patched explicitly.
- Per-patient params live in `<workspace>/reconstructed_sax_images_training_2023/segmentation/orientation_params.csv` and are produced by `reconstructed_sax_images_training_2023/compute_orientation.py` (uses the MONAI ventricular bundle in `reconstructed_sax_images_training_2023/my_models/`). The CSV covers RECON + ACDC train/test + M&Ms train/val/test (569 patients on the current data; all hit `status=ok`).
- Fallback semantics: any row with `status != "ok"` or non-finite `delta_deg` is a no-op (the loader returns the raw volume unchanged for that patient). Patients missing from the CSV are also passed through. So enabling the flag is safe even on datasets you haven't run `compute_orientation.py` for.
- Convention: target angle is π in `atan2(dy, dx)` (image-y down), i.e. RV pool ends up on the viewer's left of the LV. LV centroid is the rotation pivot and gets translated to the image centre. Flip-detection is disabled (Stage-0 QC mosaic confirmed the RECON dataset is internally consistent — no LR-mirrored cases).

**Spacing normalisation (opt-in via `--spacing_normalize`):**
- Implementation: `spacing_normalize.py` (loader side, `cv2 + numpy` only). Exposes `set_spacing_normalization(enabled, target_spacing, target_size, recon_spacing)` + `apply_to_volume(volume, spacing_xy)`. Each (T, Z) slice is resampled by `(sx/target_spacing, sy/target_spacing)` via `cv2.resize(INTER_LINEAR)`, then `_center_crop_or_pad` centres the result in a `target_size × target_size` zero-padded canvas. Output dtype is `float32`.
- **Order in the pipeline is fixed**: orientation first (operates in original-spacing coords using the precomputed CSV), spacing second. This lets the central crop be anatomically aligned with the LV — without orientation, a fixed crop would chop off the heart for patients with off-centre acquisitions.
- Spacing source per dataset:
  - **ACDC / M&Ms**: `sitk.ReadImage(path).GetSpacing()[:2]` at load time. In `data_loader.py`, NIfTI reads use `_read_nifti_4d(nii_path)` which returns `(array, (sx, sy))`. In `data_loader_rgb.py`, the spacing is read inside `load_and_orient_sitk()` and the resample is applied automatically.
  - **RECON**: no header on disk — the constant set by `--recon_spacing` (default 2.0 mm/px isotropic) is used. The two `cine = np.load(...)` sites in each loader pass `_spacing_recon_default()` explicitly.
- Loader-side bypass of the existing `aspect_preserve_resize(128, 128)`: after the spacing transform every frame is already `(128, 128)`, so the existing resize call becomes a no-op (`scale = min(128/128, 128/128) = 1`, no padding). No further changes were needed downstream.
- Percentile normalisation is still computed per-slice on the cropped output (`np.percentile(slice_seq, [1, 99])`). Less background after the crop means the p1/p99 range is tighter — usually slightly better dynamic range. Watch for it if motion thresholds need re-tuning.
- Patients with non-isotropic in-plane spacing (`sx != sy`) get a non-uniform scale factor, which the existing `cv2.resize` handles. All ACDC and M&Ms cardiac SAX scans are in-plane isotropic in practice.
- Visual QC: `python spacing_qc.py` writes `Anomaly_detection_ICCV2019/spacing_qc.png` with one row per source × three columns (raw / oriented / oriented+spacing). The heart should look the same physical size in the right column across all rows.

**ED/ES frame index conventions (per dataset):**
- **ACDC** — `Info.cfg` stores `ED` and `ES` as **1-based** indices. All loaders must subtract 1 before indexing into numpy arrays (e.g. `ed_idx = int(info['ED']) - 1`).
- **M&M** — CSV `ED` and `ES` columns are already **0-based**. Loaders use the values **as-is**, no subtraction.
- **RECON** — `segmentation/ed_es_frames.csv` (`ed_frame`, `es_frame`) is **0-based**. Used as-is.

**Validation split (ACDC test set):** `run_model.py` uses the **entire** 50-patient ACDC test set for validation (the previous 12-patient subset was not representative enough). The loader functions (`load_acdc_test_val_data`, `load_acdc_test_val_ed_es_data`) still perform the seed-42 12 / 38 patient split internally for reproducibility, but `run_model.py` concatenates both halves before passing them to the trainer.

### Dataset Paths
Datasets are expected one directory level up (`../Dataset_1`, `../Dataset_2`). Reconstructed SAX volumes are at the workspace root (`../reconstructed_sax_images_training_2023/`). Video surveillance datasets go in `../dataset/`.

### TF v1 Compatibility Note
`run_model.py` and `run_model_rgb.py` mock the `ProgressBar` module and redirect `tensorflow` to `tensorflow.compat.v1`. Any new code in these scripts must remain compatible with TF v1 graph-mode semantics. Always call `tf.compat.v1.reset_default_graph()` before building a new graph.