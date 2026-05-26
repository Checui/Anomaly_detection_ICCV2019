"""Loader-side cardiac SAX orientation normalisation.

Reads the per-patient rotation + translation parameters produced by
``reconstructed_sax_images_training_2023/compute_orientation.py`` and exposes
two helpers used by ``data_loader.py`` / ``data_loader_rgb.py``:

* ``load_orientation_params(csv_path)`` -> ``{case_id: params_dict}``
* ``apply_to_volume(volume_4d, case_id, params_map)`` -> rotated/translated copy

Kept dependency-light (only ``cv2``, ``numpy``, ``pandas``) so it can be
imported by the TF v1 training pipeline without pulling in MONAI or torch.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_PARAMS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reconstructed_sax_images_training_2023",
    "segmentation",
    "orientation_params.csv",
)


def load_orientation_params(csv_path: str | Path | None = None) -> dict[str, dict]:
    """Load the orientation CSV into ``{case_id: row_dict}``.

    Returns an empty dict (and prints a warning) if the file is missing — the
    loader treats that as a soft fail and passes volumes through unchanged.
    """
    path = Path(csv_path) if csv_path else Path(DEFAULT_PARAMS_CSV)
    if not path.exists():
        print(f"[orientation_normalize] CSV not found: {path}. Skipping normalisation.")
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[str(row["case_id"])] = row.to_dict()
    return out


def _build_affine(params: dict, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    cx_lv = float(params["lv_cx"])
    cy_lv = float(params["lv_cy"])
    delta = float(params["delta_deg"])
    flip_lr = str(params.get("flip", "none")) == "lr"
    M = cv2.getRotationMatrix2D((cx_lv, cy_lv), delta, 1.0)
    M[0, 2] += w / 2.0 - cx_lv
    M[1, 2] += h / 2.0 - cy_lv
    if flip_lr:
        flip = np.array([[-1.0, 0.0, 2.0 * cx_lv], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rot = np.vstack([M, [0.0, 0.0, 1.0]])
        M = (rot @ flip)[:2, :]
    return M


def _params_for(params_map: dict[str, dict], case_id: str) -> dict | None:
    if not params_map:
        return None
    row = params_map.get(case_id)
    if row is None:
        return None
    status = str(row.get("status", "ok"))
    if status != "ok":
        return None
    if not math.isfinite(float(row.get("delta_deg", float("nan")))):
        return None
    return row


def apply_to_frame(
    frame: np.ndarray,
    case_id: str,
    params_map: dict[str, dict],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Rotate+translate one 2D frame.  Pass-through if no usable params."""
    row = _params_for(params_map, case_id)
    if row is None:
        return frame
    h, w = frame.shape[:2]
    matrix = _build_affine(row, (h, w))
    return cv2.warpAffine(
        frame, matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_to_volume(
    volume: np.ndarray,
    case_id: str,
    params_map: dict[str, dict],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply the per-patient transform to every 2D slice of a 4D / 3D / 2D array.

    Returns the input unchanged (no copy) when no params are available so that
    enabling the feature has zero cost on missing patients.
    """
    row = _params_for(params_map, case_id)
    if row is None:
        return volume

    if volume.ndim == 2:
        return apply_to_frame(volume, case_id, params_map, interpolation)

    h, w = volume.shape[-2:]
    matrix = _build_affine(row, (h, w))

    if volume.ndim == 3:
        out = np.empty_like(volume)
        for z in range(volume.shape[0]):
            out[z] = cv2.warpAffine(
                volume[z], matrix, (w, h),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return out

    if volume.ndim == 4:
        T, Z = volume.shape[:2]
        out = np.empty_like(volume)
        for t in range(T):
            for z in range(Z):
                out[t, z] = cv2.warpAffine(
                    volume[t, z], matrix, (w, h),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        return out

    raise ValueError(f"Unsupported volume.ndim={volume.ndim}")
