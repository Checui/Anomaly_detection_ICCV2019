"""Side-by-side QC for orient + spacing normalisation.

Picks one patient per source (RECON, ACDC train, ACDC test, M&Ms train, M&Ms
val, M&Ms test) and renders three panels per row:

    raw ED mid-slice  |  after orientation  |  after orientation + spacing crop

Output: ``spacing_qc.png`` in the workspace root.

The heart should look the same physical size across patients in the right-hand
column (each panel is 128 x 128 px = 192 mm x 192 mm at the target spacing).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import orientation_normalize as orient_mod
import spacing_normalize as spacing_mod


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACDC_TRAIN = ROOT / "Dataset_2" / "database" / "training"
ACDC_TEST = ROOT / "Dataset_2" / "database" / "testing"
MM_TRAIN = ROOT / "Dataset_1" / "Training"
MM_VAL = ROOT / "Dataset_1" / "Validation"
MM_TEST = ROOT / "Dataset_1" / "Testing"
RECON_ROOT = ROOT / "reconstructed_sax_images_training_2023"
PARAMS_CSV = RECON_ROOT / "segmentation" / "orientation_params.csv"

TARGET_SPACING = 1.5
TARGET_SIZE = 128


def _load_acdc(patient_dir: Path):
    nii = patient_dir / f"{patient_dir.name}_4d.nii.gz"
    cfg = patient_dir / "Info.cfg"
    info = {}
    for line in cfg.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    ed = int(info["ED"]) - 1
    img = sitk.ReadImage(str(nii))
    arr = sitk.GetArrayFromImage(img)
    sp = tuple(float(s) for s in img.GetSpacing()[:2])
    return arr, sp, ed, patient_dir.name


def _load_mm(nii_path: Path):
    case_id = nii_path.name.replace("_sa.nii.gz", "")
    img = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(img)
    sp = tuple(float(s) for s in img.GetSpacing()[:2])
    return arr, sp, 0, case_id  # ED frame 0 is fine for visualisation


def _load_recon(npy_path: Path):
    case_id = npy_path.name.replace("_sax_recon.npy", "")
    arr = np.load(npy_path)
    return arr, (2.0, 2.0), 0, case_id


def _normalize_for_plot(frame):
    f = frame.astype(np.float32)
    lo, hi = np.percentile(f, [1, 99])
    if hi <= lo:
        return np.zeros_like(f)
    return np.clip((f - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _pick_first(directory: Path, pattern: str):
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def main():
    params_map = orient_mod.load_orientation_params(PARAMS_CSV)
    print(f"Loaded {len(params_map)} orientation params")
    spacing_mod.set_spacing_normalization(True, TARGET_SPACING, TARGET_SIZE, (2.0, 2.0))

    sources = []
    if (acdc_train_p := _pick_first(ACDC_TRAIN, "patient002")):
        sources.append(("ACDC_TRAIN", _load_acdc(acdc_train_p)))
    if (acdc_test_p := _pick_first(ACDC_TEST, "patient101")):
        sources.append(("ACDC_TEST", _load_acdc(acdc_test_p)))
    if (mm_train_p := _pick_first(MM_TRAIN, "A0S9V9_sa.nii.gz")):
        sources.append(("MM_TRAIN", _load_mm(mm_train_p)))
    if (mm_val_p := _pick_first(MM_VAL, "*_sa.nii.gz")):
        if not mm_val_p.name.endswith("_sa_gt.nii.gz"):
            sources.append(("MM_VAL", _load_mm(mm_val_p)))
    if (mm_test_p := _pick_first(MM_TEST, "*_sa.nii.gz")):
        if not mm_test_p.name.endswith("_sa_gt.nii.gz"):
            sources.append(("MM_TEST", _load_mm(mm_test_p)))
    for case_id_hint in ("P051", "P020", "P065"):
        npy = RECON_ROOT / f"{case_id_hint}_sax_recon.npy"
        if npy.exists():
            sources.append(("RECON", _load_recon(npy)))
            break

    n = len(sources)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 3.4 * n))
    axes = np.atleast_2d(axes)
    for i, (tag, (arr, spacing, ed, case_id)) in enumerate(sources):
        T, Z, H, W = arr.shape
        z = Z // 2
        raw = arr[ed, z].astype(np.float32)
        oriented = orient_mod.apply_to_volume(raw, case_id, params_map)
        normalized = spacing_mod.apply_to_volume(oriented, spacing)

        for ax, frame, title in (
            (axes[i, 0], raw,        f"{tag} {case_id}\nraw {H}x{W}  sx={spacing[0]:.3f} mm/px"),
            (axes[i, 1], oriented,   f"oriented (same px size)"),
            (axes[i, 2], normalized, f"oriented + spacing {TARGET_SIZE}x{TARGET_SIZE} = "
                                     f"{TARGET_SIZE * TARGET_SPACING:.0f} mm box"),
        ):
            ax.imshow(_normalize_for_plot(frame), cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    out_path = HERE / "spacing_qc.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
