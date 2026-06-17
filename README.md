# Cardiac MRI Anomaly Detection via Appearance–Motion Correspondence

A GAN-based **unsupervised anomaly detection** pipeline for cardiac cine MRI, adapted from
*Anomaly Detection in Video Sequence with Appearance-Motion Correspondence* (Nguyen & Meunier,
ICCV 2019). The model is trained **only on healthy (NOR) hearts** and flags cardiac pathology at
inference time as high reconstruction / prediction error — no diseased examples are needed during
training.

Originally designed for surveillance video, the architecture is repurposed here to learn the
correspondence between cardiac **appearance** (the myocardium / blood pool in a frame) and
**motion** (optical flow between phases of the heartbeat). A heart that contracts abnormally, or
that looks unlike the healthy training distribution, breaks this learned correspondence and scores
as anomalous.

> This repository is the cardiac-imaging fork of the ICCV 2019 code, developed as part of an MRes
> research project. The original surveillance-video pipeline (`main.py`) is preserved and still
> runs unchanged.

---

## How it works

The generator is a shared encoder feeding **two decoder heads**:

- An **appearance / reconstruction head** that reconstructs the input frame.
- An **auxiliary head** that predicts either the **optical flow** to the paired frame (`flow`
  model) or the paired frame itself (`rgb` model).

A PatchGAN discriminator enforces realism on the auxiliary prediction. Because the model only ever
sees healthy anatomy and healthy motion during training, an out-of-distribution (diseased) input
produces a poor reconstruction and a poor motion/frame prediction. The combination of these errors
is the **anomaly score**.

Two model variants are supported:

| `--model_type` | Auxiliary head predicts | Backend |
|---|---|---|
| `flow` (default) | Optical flow between the two frames | `GAN_tf.py` |
| `rgb` | The paired (ED) frame directly | `GAN_tf_rgb.py` |

---

## Repository layout

```
Anomaly_detection_ICCV2019/
├── run_model.py            # Main entry point — cardiac MRI training / validation
├── GAN_tf.py               # GAN with optical-flow auxiliary head (flow model)
├── GAN_tf_rgb.py           # GAN with ED-frame auxiliary head (rgb model)
├── data_loader.py          # Grayscale loader (flow pipeline): returns (images, flows)
├── data_loader_rgb.py      # RGB loader (rgb pipeline): returns (es_images, ed_images)
├── utils.py                # Anomaly scoring, patch search, AUC helpers
├── orientation_normalize.py# Optional: canonicalise heart pose (LV centred, RV on left)
├── spacing_normalize.py    # Optional: resample to fixed mm/px + centre crop
├── n4_bias_correction.py   # Optional: N4ITK bias-field correction
├── main.py                 # Original ICCV2019 surveillance-video pipeline
├── run_model.ipynb         # Interactive version of the cardiac training pipeline
├── submit_job.pbs          # Example HPC (PBS) submission script
└── requirements.txt
```

---

## Installation

Requires Python 3.10. The GAN is built on TensorFlow's v1 graph API via the compatibility shim
(`tf.compat.v1` with `tf.disable_v2_behavior()`).

```bash
git clone <your-repo-url>
cd Anomaly_detection_ICCV2019
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs tensorflow==2.12.0 and friends
```

Training logs and metrics are reported to [Weights & Biases](https://wandb.ai). Run `wandb login`
once, or set `WANDB_MODE=offline` to disable online logging.

---

## Datasets

Datasets are **not** included in this repository. The loaders expect the following public cardiac
MRI datasets, plus an optional set of reconstructed cine volumes:

| Key | Dataset | Default path | Notes |
|---|---|---|---|
| `ACDC` | [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | `--acdc_dir ../Dataset_2` | NOR cases used for training; full test set for validation |
| `MM` | [M&Ms](https://www.ub.edu/mnms/) | `--mm_dir ../Dataset_1/Training`, `--mm_val_dir ../Dataset_1/Validation` | Diagnosis CSV via `--mm_csv` |
| `RECON` | Reconstructed SAX cine `.npy` volumes (optional supplemental NOR data) | `--recon_dir ../reconstructed_sax_images_training_2023` | All NOR; ED/ES indices in `segmentation/ed_es_frames.csv` |

Validation always uses datasets that contain pathology (`ACDC` / `MM`); `RECON` is training-only
because every RECON subject is healthy and so cannot supply positive (anomalous) samples for AUC.

**ED/ES index conventions** (handled internally, noted here for reference): ACDC `Info.cfg`
indices are 1-based; M&Ms CSV indices and the RECON CSV are 0-based.

---

## Usage

`run_model.py` is the unified entry point. Three orthogonal flags control every run:

### 1. `--model_type` — auxiliary head
`flow` (predict optical flow, default) or `rgb` (predict the paired frame).

### 2. `--datasets` — training data
One or more of `ACDC MM RECON` (space-separated). `--recon_dir` is required when `RECON` is used.

### 3. `--frame_mode` — which frame pairs to use

| Mode | Input frame | Auxiliary target |
|---|---|---|
| `ed_es` (default) | ES | flow ES→ED / predict ED |
| `es_ed` | ED | flow ED→ES / predict ES (inverse of `ed_es`) |
| `next_frame` | every frame | the next consecutive frame |
| `next_frame_systole` | frames in the systolic window | the next consecutive frame |

`next_frame_systole` restricts consecutive pairs to the contraction phase `t ∈ [ED, ES−1]`.

### Quick start

```bash
# Main experiment: flow model, ACDC + M&Ms, ED/ES pairs, 100 epochs
python run_model.py --model_type flow --datasets ACDC MM --frame_mode ed_es --epochs 100 \
    --acdc_dir ../Dataset_2 \
    --mm_dir ../Dataset_1/Training \
    --mm_val_dir ../Dataset_1/Validation \
    --mm_csv "../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
```

```bash
# RGB model, all three datasets
python run_model.py --model_type rgb --datasets ACDC MM RECON --frame_mode ed_es --epochs 100 \
    --recon_dir ../reconstructed_sax_images_training_2023 \
    --acdc_dir ../Dataset_2 --mm_dir ../Dataset_1/Training \
    --mm_val_dir ../Dataset_1/Validation \
    --mm_csv "../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
```

```bash
# Resume from a saved checkpoint (start_epoch must match a saved checkpoint number)
python run_model.py ... --start_epoch 30 --epochs 100
```

The checkpoint folder name encodes the run configuration, e.g. `ACDC_MM_ED_ES_FLOW_NOR`.
Checkpoints are written to `training_saver/<name>/` and sample images to `generated/<name>/`.

### Loss weights (optional)

The generator loss is `G_loss_total = lw_adv·G_adv + lw_appe·loss_appe + lw_aux·loss_aux`.
Defaults reproduce the ICCV 2019 ratios:

| Flag | Default | Term |
|---|---|---|
| `--lw_adv`  | `0.25` | Adversarial (GAN) term |
| `--lw_appe` | `1.0`  | Appearance reconstruction loss |
| `--lw_aux`  | `2.0`  | Auxiliary prediction loss (flow or ED) |

```bash
# Ablation: upweight the appearance term
python run_model.py ... --lw_appe 2.0 --lw_aux 1.0
```

### Preprocessing (all optional, off by default)

These run in a fixed order in the loader pipeline: **N4 bias correction → orientation → spacing →
per-slice percentile normalisation.** Each flag is a safe no-op when its required inputs are
missing, so you can enable them incrementally.

**Orientation normalisation** — canonicalise the heart pose (LV centroid centred, RV pool on the
viewer's left) per patient. Requires a precomputed `orientation_params.csv`.

```bash
python run_model.py ... --orient_normalize \
    --orient_params ../reconstructed_sax_images_training_2023/segmentation/orientation_params.csv
```

**Spacing normalisation** — resample to a fixed in-plane resolution and centre-crop to a fixed
size (default 1.5 mm/px × 128 px → 192 mm FoV). Applied after orientation so the crop stays
anatomically centred.

```bash
python run_model.py ... --orient_normalize --spacing_normalize \
    --target_spacing 1.5 --target_size 128 --recon_spacing 2.0
```

**N4ITK bias-field correction** — correct intensity inhomogeneity. The field is estimated once per
(patient, z-slice) from the temporal-mean frame and shared across all frames, so frame-to-frame
motion (and hence optical flow) is preserved.

```bash
python run_model.py ... --n4_bias_correct --n4_shrink 4 --n4_iterations 50 --n4_levels 4
```

### HPC submission (PBS)

```bash
qsub submit_job.pbs
```

> Note: `submit_job.pbs` still references an older `--dataset COMBINED` syntax — update it to the
> current `--datasets` / `--frame_mode` flags before submitting.

---

## Architecture

**Generator.** A shared Inception-style encoder followed by two independent decoder heads:

- **Auxiliary decoder** (U-Net skip connections from the encoder) predicts optical flow (`flow`)
  or the ED frame (`rgb`). The flow head uses a **linear** output (GT flow is raw pixel
  displacement, not in `[-1, 1]`, so `tanh` would saturate); the ED head uses `tanh` because its
  target is already in `[-1, 1]`.
- **Reconstruction decoder** (no skip connections) reconstructs the input frame, bounded by
  `tanh`. Dropping skip connections forces a full bottleneck pass, making reconstruction harder for
  out-of-distribution inputs — which is exactly the behaviour anomaly detection relies on.

Decoder blocks follow `deconv → BN → leaky_relu → dropout → concat_skip`.

**Discriminator.** PatchGAN-style, taking the concatenation of the true frame and the predicted
flow/frame.

## Evaluation

At every validation epoch the model is scored against held-out data containing pathology, and
several AUROC pipelines are logged to W&B:

- **Full-frame score** — the ICCV 2019 combined error
  `log(loss_appe/μ_appe) + 2·log(loss_aux/μ_aux)`, with guards against `log(0)`.
- **Patch-based score** — a 16×16 patch (stride 4) with the highest auxiliary error is selected
  (paper §3.5), with appearance error read at the same location.
- **Optimal-weight search** — sweeps the appearance weight over `[0, 1]` to log an upper-bound
  reference AUC.
- **Per-disease one-vs-NOR AUC** — separate AUROC for each pathology (MINF, DCM, HCM, RV).
- **Patient-level AUC** — per-sample scores aggregated per patient with five aggregators (Mean,
  FrameMax, FrameTop20, SliceMax, SliceTop20).

The normalising means `μ_*` are recomputed each validation epoch over the entire training set in
eval mode.

---

## Original surveillance-video pipeline

The original ICCV 2019 video anomaly detection workflow is preserved in `main.py`:

```bash
python main.py -d UCSDped2 -t 1        # prepare data
python main.py -d UCSDped2 -t 2 -e 40  # train
python main.py -d UCSDped2 -t 3 -c 0   # test one clip
python main.py -d UCSDped2 -t 4        # evaluate AUC
```

Available datasets: `UCSDped2`, `Avenue`, `Belleview`, `Train`, `Exit`, `Entrance`. Surveillance
datasets are expected under `../dataset/`.

---

## Acknowledgements & citation

This work builds directly on the architecture and reference implementation from:

> Trong-Nguyen Nguyen and Jean Meunier.
> **Anomaly Detection in Video Sequence with Appearance-Motion Correspondence.**
> *IEEE International Conference on Computer Vision (ICCV), 2019.*
> [paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Anomaly_Detection_in_Video_Sequence_With_Appearance-Motion_Correspondence_ICCV_2019_paper.html)
> · [arXiv](https://arxiv.org/pdf/1908.06351.pdf)
> · [demo](https://youtu.be/PaUenXHHzuw)

```bibtex
@InProceedings{Nguyen_2019_ICCV,
  author    = {Nguyen, Trong-Nguyen and Meunier, Jean},
  title     = {Anomaly Detection in Video Sequence With Appearance-Motion Correspondence},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2019}
}
```

If you use the cardiac MRI adaptation, please also link back to this repository.

## License

Released under the **BSD 2-Clause License** (see [`LICENSE`](./LICENSE)), inherited from the
original ICCV 2019 implementation. © 2020 Trong-Nguyen Nguyen.
