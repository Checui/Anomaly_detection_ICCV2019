"""Loader-side N4ITK bias-field correction.

Corrects the low-frequency multiplicative intensity inhomogeneity ("bias
field") that MRI scanners introduce through B1 / receive-coil non-uniformity.
Designed to run as the FIRST loader step, on the raw acquired intensities,
before orientation / spacing / percentile normalisation.

Why estimate the field once per slice and share it across time
--------------------------------------------------------------
For a cine acquisition the slice location is fixed in the scanner frame; only
the anatomy moves within it.  The bias field is a property of the scanner +
coil, so for a given (patient, z-slice) it is the same across every cardiac
phase.  We therefore estimate the field once from the temporal-mean frame
(higher SNR) and divide all T frames of that slice by the same field.  This is
both ~T times cheaper than per-frame N4 and, crucially for the optical-flow
head, it preserves frame-to-frame intensity dynamics: a per-frame field would
inject spurious brightness changes that Farneback flow would read as motion.

Per-2D-slice (rather than full-3D) correction matches the existing per-slice
percentile normalisation downstream, so any residual inter-slice brightness
offset is washed out by that step.

Dependency: SimpleITK (already a hard dependency of both data loaders) + numpy.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


# Module-level config, toggled by ``set_n4_bias_correction``.
_ENABLED = False
_SHRINK = 4               # downsample factor for the field-estimation grid
_ITERS = 50               # max iterations per fitting level
_LEVELS = 4               # number of B-spline fitting levels
_HIST_BINS = 200          # bins for the Otsu foreground mask
_EPS = 1e-6               # floor on the bias field before dividing


def set_n4_bias_correction(
    enabled: bool,
    shrink_factor: int = 4,
    n_iterations: int = 50,
    n_fitting_levels: int = 4,
    n_histogram_bins: int = 200,
) -> None:
    """Toggle N4 bias-field correction. Call BEFORE invoking any load_* function."""
    global _ENABLED, _SHRINK, _ITERS, _LEVELS, _HIST_BINS
    _ENABLED = bool(enabled)
    _SHRINK = max(1, int(shrink_factor))
    _ITERS = max(1, int(n_iterations))
    _LEVELS = max(1, int(n_fitting_levels))
    _HIST_BINS = max(2, int(n_histogram_bins))


def is_enabled() -> bool:
    return _ENABLED


def _effective_shrink(h: int, w: int) -> int:
    """Clamp the shrink factor so the estimation grid stays >= ~16 px per side."""
    eff = _SHRINK
    while eff > 1 and (min(h, w) // eff) < 16:
        eff -= 1
    return eff


def _estimate_bias_field(ref: np.ndarray):
    """Estimate the multiplicative bias field for one 2D frame.

    Returns a ``(H, W)`` float32 array (the field, floored at ``_EPS``) or
    ``None`` when the slice is degenerate / N4 fails, in which case the caller
    leaves the frame(s) unchanged.
    """
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if ref.ndim != 2 or ref.size == 0 or float(ref.max()) <= float(ref.min()):
        return None

    img = sitk.Cast(sitk.GetImageFromArray(ref), sitk.sitkFloat32)

    try:
        mask = sitk.OtsuThreshold(img, 0, 1, _HIST_BINS)  # background=0, foreground=1
    except Exception:
        return None
    if int(np.asarray(sitk.GetArrayViewFromImage(mask)).sum()) == 0:
        return None  # no foreground to correct

    h, w = ref.shape
    eff = _effective_shrink(h, w)
    est_img, est_mask = img, mask
    if eff > 1:
        try:
            est_img = sitk.Shrink(img, [eff, eff])
            est_mask = sitk.Shrink(mask, [eff, eff])
        except Exception:
            est_img, est_mask = img, mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([_ITERS] * _LEVELS)
    try:
        corrector.Execute(est_img, est_mask)
        # Evaluate the log bias field at full resolution, then exponentiate.
        log_bias = corrector.GetLogBiasFieldAsImage(img)
    except Exception:
        return None

    bias = np.asarray(sitk.GetArrayFromImage(sitk.Exp(log_bias)), dtype=np.float32)
    if bias.shape != ref.shape or not np.isfinite(bias).all():
        return None
    return np.maximum(bias, _EPS)


def _correct_2d(frame: np.ndarray) -> np.ndarray:
    bias = _estimate_bias_field(frame)
    if bias is None:
        return frame.astype(np.float32, copy=False)
    return (frame.astype(np.float32) / bias).astype(np.float32)


def _correct_slice_stack(stack: np.ndarray) -> np.ndarray:
    """Correct a ``(T, H, W)`` temporal stack with a single shared field."""
    bias = _estimate_bias_field(stack.mean(axis=0))
    if bias is None:
        return stack.astype(np.float32, copy=False)
    return (stack.astype(np.float32) / bias[None, :, :]).astype(np.float32)


def apply_to_volume(volume: np.ndarray) -> np.ndarray:
    """N4-correct a 2D / 3D / 4D array. Pass-through if disabled or ``None``.

    * 2D ``(H, W)``        — single-frame correction.
    * 3D ``(Z, H, W)``     — each slice corrected independently.
    * 4D ``(T, Z, H, W)``  — per (z) slice, one field estimated from the temporal
      mean and applied to all T frames (preserves temporal dynamics).
    """
    if not _ENABLED or volume is None:
        return volume

    arr = np.asarray(volume)
    if arr.ndim == 2:
        return _correct_2d(arr)

    if arr.ndim == 3:
        out = np.empty(arr.shape, dtype=np.float32)
        for z in range(arr.shape[0]):
            out[z] = _correct_2d(arr[z])
        return out

    if arr.ndim == 4:
        T, Z = arr.shape[:2]
        out = np.empty(arr.shape, dtype=np.float32)
        for z in range(Z):
            out[:, z] = _correct_slice_stack(arr[:, z])
        return out

    raise ValueError(f"Unsupported volume.ndim={arr.ndim}")
