"""Loader-side spacing normalisation.

Resamples a 2D / 3D / 4D cardiac SAX volume from its native millimetre spacing
to a fixed target spacing, then centre-crops (zero-pads) to a fixed pixel size
so every patient ends up with the same in-plane physical size per pixel and
the same field of view.

Designed to run AFTER ``orientation_normalize.apply_to_volume`` — the LV is
already centred, so the central crop is anatomically meaningful (the heart
sits where we crop, not at a random offset).

Dependency-light: only ``cv2`` + ``numpy``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# Module-level config, toggled by ``set_spacing_normalization``.
_ENABLED = False
_TARGET_SPACING = 1.5            # mm/px
_TARGET_SIZE = 128               # output side length in px (128 -> 192 mm FoV at 1.5 mm/px)
_RECON_SPACING = (2.0, 2.0)      # default spacing for RECON .npy volumes (no header on disk)


def set_spacing_normalization(
    enabled: bool,
    target_spacing: float = 1.5,
    target_size: int = 128,
    recon_spacing=(2.0, 2.0),
) -> None:
    """Toggle spacing normalisation. Call BEFORE invoking any load_* function."""
    global _ENABLED, _TARGET_SPACING, _TARGET_SIZE, _RECON_SPACING
    _ENABLED = bool(enabled)
    _TARGET_SPACING = float(target_spacing)
    _TARGET_SIZE = int(target_size)
    _RECON_SPACING = (float(recon_spacing[0]), float(recon_spacing[1]))


def is_enabled() -> bool:
    return _ENABLED


def target_size() -> int:
    return _TARGET_SIZE


def recon_spacing() -> tuple[float, float]:
    return _RECON_SPACING


def _resample_one_frame(frame: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    h, w = frame.shape
    new_h = max(1, int(round(h * scale_y)))
    new_w = max(1, int(round(w * scale_x)))
    if (new_h, new_w) == (h, w):
        return frame.astype(np.float32, copy=False)
    return cv2.resize(
        frame.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )


def _center_crop_or_pad(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape
    out = np.zeros((target, target), dtype=np.float32)
    # source region (centred on the input)
    src_y0 = max(0, (h - target) // 2)
    src_x0 = max(0, (w - target) // 2)
    src_h = min(h, target)
    src_w = min(w, target)
    # destination region (centred on the output)
    dst_y0 = max(0, (target - h) // 2)
    dst_x0 = max(0, (target - w) // 2)
    out[dst_y0 : dst_y0 + src_h, dst_x0 : dst_x0 + src_w] = (
        frame[src_y0 : src_y0 + src_h, src_x0 : src_x0 + src_w]
    )
    return out


def _transform_frame(frame: np.ndarray, scale_x: float, scale_y: float, target: int) -> np.ndarray:
    return _center_crop_or_pad(_resample_one_frame(frame, scale_x, scale_y), target)


def apply_to_volume(volume: np.ndarray, spacing_xy) -> np.ndarray:
    """Resample + central crop every (T, Z) slice of a 4D array.

    Also accepts 3D (Z, H, W) and 2D (H, W) arrays.  Pass-through if disabled.
    ``spacing_xy`` is the (sx, sy) tuple in millimetres per pixel.  For RECON
    volumes that lack a header, pass ``recon_spacing()`` or supply your own.
    """
    if not _ENABLED or volume is None:
        return volume
    sx, sy = float(spacing_xy[0]), float(spacing_xy[1])
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        return volume
    scale_x = sx / _TARGET_SPACING
    scale_y = sy / _TARGET_SPACING
    target = _TARGET_SIZE

    if volume.ndim == 2:
        return _transform_frame(volume, scale_x, scale_y, target)

    if volume.ndim == 3:
        out = np.empty((volume.shape[0], target, target), dtype=np.float32)
        for i in range(volume.shape[0]):
            out[i] = _transform_frame(volume[i], scale_x, scale_y, target)
        return out

    if volume.ndim == 4:
        T, Z = volume.shape[:2]
        out = np.empty((T, Z, target, target), dtype=np.float32)
        for t in range(T):
            for z in range(Z):
                out[t, z] = _transform_frame(volume[t, z], scale_x, scale_y, target)
        return out

    raise ValueError(f"Unsupported volume.ndim={volume.ndim}")
