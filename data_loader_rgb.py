import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import random

# ── Orientation normalisation (opt-in) ──────────────────────────────────────
try:
    from orientation_normalize import (
        apply_to_volume as _orient_apply_to_volume,
        load_orientation_params as _orient_load_params,
    )
    _ORIENT_AVAILABLE = True
except ImportError:
    _ORIENT_AVAILABLE = False
    def _orient_apply_to_volume(volume, case_id, params_map):
        return volume
    def _orient_load_params(csv_path=None):
        return {}

_ORIENT_PARAMS_MAP: dict = {}
_ORIENT_ENABLED = False


def set_orientation_normalization(enabled, csv_path=None):
    """Toggle orientation normalisation. Call before any load_* function."""
    global _ORIENT_PARAMS_MAP, _ORIENT_ENABLED
    _ORIENT_ENABLED = bool(enabled) and _ORIENT_AVAILABLE
    _ORIENT_PARAMS_MAP = _orient_load_params(csv_path) if _ORIENT_ENABLED else {}
    if enabled and not _ORIENT_AVAILABLE:
        print("[data_loader_rgb] orientation_normalize unavailable; skipping rotation.")


def _maybe_normalize_volume(volume, case_id):
    if not _ORIENT_ENABLED or volume is None:
        return volume
    return _orient_apply_to_volume(volume, case_id, _ORIENT_PARAMS_MAP)


# ── Spacing normalisation (opt-in) ──────────────────────────────────────────
try:
    from spacing_normalize import (
        apply_to_volume as _spacing_apply_to_volume,
        is_enabled as _spacing_is_enabled,
        recon_spacing as _spacing_recon_default,
        set_spacing_normalization as _spacing_set_normalization,
    )
    _SPACING_AVAILABLE = True
except ImportError:
    _SPACING_AVAILABLE = False
    def _spacing_apply_to_volume(volume, spacing_xy):
        return volume
    def _spacing_is_enabled():
        return False
    def _spacing_recon_default():
        return (2.0, 2.0)
    def _spacing_set_normalization(*args, **kwargs):
        pass


def set_spacing_normalization(enabled, target_spacing=1.5, target_size=128,
                              recon_spacing=(2.0, 2.0)):
    """Toggle spacing normalisation. Call before any load_* function."""
    if not _SPACING_AVAILABLE:
        if enabled:
            print("[data_loader_rgb] spacing_normalize unavailable; skipping resample.")
        return
    _spacing_set_normalization(enabled, target_spacing, target_size, recon_spacing)


def _maybe_resample_volume(volume, spacing_xy):
    if not _spacing_is_enabled() or volume is None:
        return volume
    return _spacing_apply_to_volume(volume, spacing_xy)


# ── ED/ES extraction direction ──────────────────────────────────────────────
#
# Controls which cardiac phase is the model INPUT when extracting ED/ES pairs.
#   'es' (default): input = ES frame, reconstruction target = ED frame.  This is
#                   the original cardiac setup (the "ed_es" frame mode).
#   'ed'          : inverse — input = ED frame, target = ES frame (the "es_ed"
#                   frame mode).
# Set once per run by run_model.py via set_edes_direction().
_EDES_INPUT_PHASE = 'es'


def set_edes_direction(input_phase):
    """Select which phase is the model input for ED/ES pair extraction.

    input_phase='es' (default): input = ES frame, target = ED frame.
    input_phase='ed'          : input = ED frame, target = ES frame.
    Any other value falls back to 'es'.
    """
    global _EDES_INPUT_PHASE
    _EDES_INPUT_PHASE = 'ed' if str(input_phase).lower() == 'ed' else 'es'


def _edes_input_target(ed_rgb, es_rgb):
    """Return (input_rgb, target_rgb) honouring the ED/ES direction.

    Default ('es'): input ES, target ED.  Inverse ('ed'): input ED, target ES.
    """
    if _EDES_INPUT_PHASE == 'ed':
        return ed_rgb, es_rgb
    return es_rgb, ed_rgb


def _middle_slice_range(Z, frac=0.2):
    """Range covering the middle (1 - 2*frac) of Z slices.

    Drops the top and bottom ``frac`` of slices (default 20% each, keeping the
    middle 60%).  For small Z, drops at least 1 slice from each end so the
    behaviour matches the previous ``range(1, Z - 1)`` convention.
    """
    n_drop = max(1, int(round(frac * Z)))
    stop = max(n_drop, Z - n_drop)
    return range(n_drop, stop)


# =============================================================================
# 1. CORE IMAGE UTILITIES
# =============================================================================

def _case_id_from_nii_path(nii_path):
    """Derive case_id from filename if it looks like an image (not a _gt mask)."""
    name = os.path.basename(str(nii_path))
    if name.endswith("_gt.nii.gz"):
        return None
    for suffix in ("_sa.nii.gz", "_4d.nii.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def load_and_orient_sitk(nii_path):
    """Safely loads 4D NIfTI images WITHOUT breaking the oblique SAX plane.

    When orientation normalisation is enabled (set_orientation_normalization),
    cine images are auto-rotated/translated based on the case_id derived from
    the filename.  Mask files (``*_gt.nii.gz``) are passed through unchanged.

    When spacing normalisation is enabled (set_spacing_normalization), cine
    images are additionally resampled to the target millimetre spacing and
    centre-cropped to the target pixel size.  Masks are passed through.
    """
    image = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(image)
    case_id = _case_id_from_nii_path(nii_path)
    if case_id is not None:
        arr = _maybe_normalize_volume(arr, case_id)
        spacing_xy = tuple(float(s) for s in image.GetSpacing()[:2])
        arr = _maybe_resample_volume(arr, spacing_xy)
    return arr

def center_crop_or_pad(image, target_h=128, target_w=128):
    """Center crops an image, padding with zeros if it's smaller than the target size."""
    h, w = image.shape
    pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image, 
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), 
            mode='constant', constant_values=0
        )
        h, w = image.shape

    center_y, center_x = h // 2, w // 2
    start_y, start_x = center_y - target_h // 2, center_x - target_w // 2
    return image[start_y:start_y + target_h, start_x:start_x + target_w]

def aspect_preserve_resize(image, target_h=128, target_w=128):
    """Resizes image to fit within target dimensions while preserving aspect ratio, then pads."""
    h, w = image.shape
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_h, pad_w = target_h - new_h, target_w - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def normalize_frame_to_rgb(frame, p1, p99, center_y, center_x, target_size):
    """Crops around the center of mass and normalizes a 2D frame."""
    # Use our new cropping logic
    # frame_cropped = crop_around_center(frame.astype(np.float32), center_y, center_x, *target_size)
    # Use aspect preserve resize
    frame_resized = aspect_preserve_resize(frame.astype(np.float32), *target_size)
    
    if (p99 - p1) < 1e-7:
        frame_norm = np.zeros(target_size, dtype=np.float32)
    else:
        frame_norm = np.clip((frame_resized - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        
    return np.stack([frame_norm] * 3, axis=-1)


def get_slice_centers_from_mask(mask_arr):
    """Calculates a single, fixed (center_y, center_x) for each Z-slice."""
    # If M&M 4D mask (T, Z, H, W), collapse the Time dimension to get a union of all masks
    if mask_arr.ndim == 4:
        mask_collapsed = np.max(mask_arr, axis=0) 
    else:
        # If ACDC 3D mask (Z, H, W)
        mask_collapsed = mask_arr
        
    Z, H, W = mask_collapsed.shape
    centers = []
    
    for z in range(Z):
        y_indices, x_indices = np.where(mask_collapsed[z] > 0)
        if len(y_indices) > 0:
            cy = int(np.round(np.mean(y_indices)))
            cx = int(np.round(np.mean(x_indices)))
            centers.append((cy, cx))
        else:
            # Fallback to absolute image center if no heart is found in this slice
            centers.append((H // 2, W // 2))
            
    return centers

def crop_around_center(image, center_y, center_x, target_h=128, target_w=128):
    """Crops and pads an image evenly around a specific coordinate."""
    h, w = image.shape
    half_h, half_w = target_h // 2, target_w // 2
    
    start_y, end_y = center_y - half_h, center_y + half_h
    start_x, end_x = center_x - half_w, center_x + half_w
    
    # Pad with zeros if the bounding box goes outside the image dimensions
    pad_top = max(0, -start_y)
    pad_bottom = max(0, end_y - h)
    pad_left = max(0, -start_x)
    pad_right = max(0, end_x - w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        # Shift starting coordinates because the image just grew
        start_y += pad_top; end_y += pad_top
        start_x += pad_left; end_x += pad_left
        
    return image[start_y:end_y, start_x:end_x]

# =============================================================================
# 2. 4D SEQUENCE PROCESSORS (The core logic extracted)
# =============================================================================

def extract_consecutive_pairs(img_arr, mask_arr, target_size=(128, 128), return_slice_idxs=False):
    """Extracts every (t, t+1) frame pair from a 4D array.

    If return_slice_idxs is True, also returns a list of per-sample z indices.
    """
    if img_arr.ndim != 4:
        return ([], [], []) if return_slice_idxs else ([], [])

    T, Z, _, _ = img_arr.shape
    images, target_frames, slice_idxs = [], [], []

    for z in _middle_slice_range(Z):
        slice_seq = img_arr[:, z, :, :]
        p1, p99 = np.percentile(slice_seq, 1), np.percentile(slice_seq, 99)

        processed = [normalize_frame_to_rgb(slice_seq[t], p1, p99, 0, 0, target_size) for t in range(T)]

        for t in range(T - 1):
            images.append(processed[t])
            target_frames.append(processed[t+1])
            slice_idxs.append(z)

    if return_slice_idxs:
        return images, target_frames, slice_idxs
    return images, target_frames

def extract_consecutive_pairs_systole(img_arr, mask_arr, ed_idx, es_idx,
                                       target_size=(128, 128), return_slice_idxs=False):
    """Extract every (t, t+1) frame pair with t in [ed_idx, es_idx - 1].

    Restricts emission to the systolic contraction phase (ED→ES).  Returns
    empty lists if es_idx <= ed_idx or max(ed_idx, es_idx) >= T.
    """
    if img_arr.ndim != 4:
        return ([], [], []) if return_slice_idxs else ([], [])

    T, Z, _, _ = img_arr.shape
    if es_idx <= ed_idx or es_idx >= T or ed_idx < 0:
        return ([], [], []) if return_slice_idxs else ([], [])

    images, target_frames, slice_idxs = [], [], []

    for z in _middle_slice_range(Z):
        slice_seq = img_arr[:, z, :, :]
        p1, p99 = np.percentile(slice_seq, 1), np.percentile(slice_seq, 99)

        processed = [normalize_frame_to_rgb(slice_seq[t], p1, p99, 0, 0, target_size)
                     for t in range(T)]

        for t in range(ed_idx, es_idx):
            images.append(processed[t])
            target_frames.append(processed[t + 1])
            slice_idxs.append(z)

    if return_slice_idxs:
        return images, target_frames, slice_idxs
    return images, target_frames

def extract_edes_pairs(img_arr,mask_arr, ed_idx, es_idx, target_size=(128, 128), motion_threshold=0.01, return_slice_idxs=False):
    if img_arr.ndim != 4:
        return ([], [], []) if return_slice_idxs else ([], [])

    T, Z, _, _ = img_arr.shape
    if max(ed_idx, es_idx) >= T or ed_idx == es_idx:
        return ([], [], []) if return_slice_idxs else ([], [])

    # GET CENTERS FOR ALL Z-SLICES ONCE
    # centers = get_slice_centers_from_mask(mask_arr)
    images, target_frames, slice_idxs = [], [], []

    for z in _middle_slice_range(Z):
        # cy, cx = centers[z] # Unpack the center for this specific slice

        slice_seq = img_arr[:, z, :, :]
        p1, p99 = np.percentile(slice_seq, 1), np.percentile(slice_seq, 99)

        # Pass cy and cx into the normalizer
        # es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, cy, cx, target_size)
        # ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, cy, cx, target_size)
        # resize
        es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, 0, 0, target_size)
        ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, 0, 0, target_size)
        input_rgb, target_rgb = _edes_input_target(ed_rgb, es_rgb)

        if np.mean(np.abs(ed_rgb - es_rgb)) >= motion_threshold:
            images.append(input_rgb)
            target_frames.append(target_rgb)
            slice_idxs.append(z)

    if return_slice_idxs:
        return images, target_frames, slice_idxs
    return images, target_frames

def read_acdc_info(cfg_path):
    """Safely reads ACDC Info.cfg files into a dictionary."""
    info = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.strip().split(':', 1)
                        info[key.strip()] = val.strip()
        except Exception:
            pass
    return info

def _to_array(lst):
    return np.array(lst) if len(lst) > 0 else np.array([])

# =============================================================================
# 3. DATASET LOADERS (Corrected)
# =============================================================================

def load_acdc_data(base_dir, target_size=(128, 128), restrict_to_systole=False):
    training_dir = os.path.join(base_dir, 'database', 'training')
    patients = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])

    all_images, all_targets = [], []
    print(f"Found {len(patients)} patients in ACDC.")

    for p in patients:
        p_dir = os.path.join(training_dir, p)
        info = read_acdc_info(os.path.join(p_dir, 'Info.cfg'))

        if info.get('Group', '') != 'NOR':
            continue

        ed_idx = es_idx = None
        if restrict_to_systole:
            try:
                # ACDC Info.cfg uses 1-based frame indices; subtract 1 for 0-based numpy indexing
                ed_idx = int(info['ED']) - 1
                es_idx = int(info['ES']) - 1
            except (KeyError, ValueError) as e:
                print(f"Skipping ACDC {p}: missing ED/ES ({e})")
                continue
            if es_idx <= ed_idx:
                print(f"Skipping ACDC {p}: ES ({es_idx}) <= ED ({ed_idx})")
                continue

        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            if restrict_to_systole:
                imgs, targets = extract_consecutive_pairs_systole(
                    img_arr, 0, ed_idx, es_idx, target_size)
                if not imgs:
                    print(f"Skipping ACDC {p}: ES ({es_idx}) out of range or no pairs")
            else:
                imgs, targets = extract_consecutive_pairs(img_arr, 0, target_size)

            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error loading {p}: {e}")

    return _to_array(all_images), _to_array(all_targets)


def load_mm_data(mm_training_dir, csv_path, target_size=(128, 128), restrict_to_systole=False):
    df = pd.read_csv(csv_path)
    nor_rows = df[df['Pathology'] == 'NOR'][['External code', 'ED', 'ES']]
    nor_info = {row['External code']: (int(row['ED']), int(row['ES']))
                for _, row in nor_rows.iterrows()}
    print(f"Found {len(nor_info)} M&M NOR subjects in CSV.")

    all_images, all_targets = [], []
    sa_files = sorted([f for f in os.listdir(mm_training_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in nor_info:
            continue

        ed_idx = es_idx = None
        if restrict_to_systole:
            ed_idx, es_idx = nor_info[subject_id]
            if es_idx <= ed_idx:
                print(f"Skipping M&M {subject_id}: ES ({es_idx}) <= ED ({ed_idx})")
                continue

        nii_path = os.path.join(mm_training_dir, fname)
        if not os.path.exists(nii_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            if restrict_to_systole:
                imgs, targets = extract_consecutive_pairs_systole(
                    img_arr, 0, ed_idx, es_idx, target_size)
                if not imgs:
                    print(f"Skipping M&M {subject_id}: ES ({es_idx}) out of range or no pairs")
            else:
                imgs, targets = extract_consecutive_pairs(img_arr, 0, target_size)
            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")

    return _to_array(all_images), _to_array(all_targets)


def load_combined_data(acdc_dir, mm_training_dir, csv_path, target_size=(128, 128)):
    print("=== Loading ACDC NOR data ===")
    acdc_images, acdc_targets = load_acdc_data(acdc_dir, target_size)
    
    print("\n=== Loading M&M NOR data ===")
    mm_images, mm_targets = load_mm_data(mm_training_dir, csv_path, target_size)
    
    images = _to_array(list(acdc_images) + list(mm_images))
    targets = _to_array(list(acdc_targets) + list(mm_targets))
    print(f"\nCombined: {len(images)} samples total")
    
    return images, targets


def load_combined_ed_es_data(acdc_dir, mm_training_dir, csv_path, target_size=(128, 128)):
    all_images, all_targets = [], []
    
    # --- ACDC ---
    training_dir = os.path.join(acdc_dir, 'database', 'training')
    if not os.path.isdir(training_dir): 
        training_dir = os.path.join(acdc_dir, 'database', 'training_test')
    
    patients = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])
    for p in patients:
        p_dir = os.path.join(training_dir, p)
        info = read_acdc_info(os.path.join(p_dir, 'Info.cfg'))
        
        if info.get('Group', '') != 'NOR' or 'ED' not in info or 'ES' not in info:
            continue
            
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        ed_frame_str = str(info['ED']).zfill(2)
        mask_path = os.path.join(p_dir, f'{p}_frame{ed_frame_str}_gt.nii.gz')
        
        if not os.path.exists(nii_path) or not os.path.exists(mask_path): 
            continue
            
        try:
            img_arr = load_and_orient_sitk(nii_path)
            mask_arr = load_and_orient_sitk(mask_path)
            # FIXED: Pass mask_arr into the extractor!
            # imgs, targets = extract_edes_pairs(img_arr, mask_arr, int(info['ED']), int(info['ES']), target_size)
            # ACDC Info.cfg uses 1-based frame indices; subtract 1 for 0-based numpy indexing
            imgs, targets = extract_edes_pairs(img_arr, 0, int(info['ED']) - 1, int(info['ES']) - 1, target_size)
            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error processing ACDC {p}: {e}")

    acdc_count = len(all_images)
    print(f"ACDC NOR ED/ES samples: {acdc_count}")

    # --- M&M ---
    df = pd.read_csv(csv_path)
    nor_rows = df[df['Pathology'] == 'NOR'][['External code', 'ED', 'ES']]
    nor_info = {row['External code']: (int(row['ED']), int(row['ES'])) for _, row in nor_rows.iterrows()}
    
    sa_files = sorted([f for f in os.listdir(mm_training_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])
    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in nor_info: continue
        
        nii_path = os.path.join(mm_training_dir, fname)
        mask_path = os.path.join(mm_training_dir, fname.replace('_sa.nii.gz', '_sa_gt.nii.gz'))
        
        if not os.path.exists(nii_path) or not os.path.exists(mask_path): 
            continue
            
        try:
            ed_idx, es_idx = nor_info[subject_id]
            img_arr = load_and_orient_sitk(nii_path)
            # mask_arr = load_and_orient_sitk(mask_path)
            # FIXED: Pass mask_arr into the extractor!
            # imgs, targets = extract_edes_pairs(img_arr, mask_arr, ed_idx, es_idx, target_size)
            # resize
            imgs, targets = extract_edes_pairs(img_arr, 0, ed_idx, es_idx, target_size)
            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error processing M&M {subject_id}: {e}")

    print(f"M&M NOR ED/ES samples: {len(all_images) - acdc_count}")
    return _to_array(all_images), _to_array(all_targets)


def load_acdc_test_val_data(base_dir, target_size=(128, 128), seed=42, restrict_to_systole=False):
    testing_dir = os.path.join(base_dir, 'database', 'testing')
    all_patient_dirs = sorted([d for d in os.listdir(testing_dir) if os.path.isdir(os.path.join(testing_dir, d))])

    group_patients, patient_info = {}, {}
    for p in all_patient_dirs:
        info = read_acdc_info(os.path.join(testing_dir, p, 'Info.cfg'))
        if group := info.get('Group'):
            patient_info[p] = info
            group_patients.setdefault(group, []).append(p)

    rng = random.Random(seed)
    val_set, test_set = set(), set()
    val_counts = {'NOR': 4, 'MINF': 2, 'DCM': 2, 'HCM': 2, 'RV': 2}

    for group, pats in sorted(group_patients.items()):
        shuffled = pats[:]
        rng.shuffle(shuffled)
        n_val = val_counts.get(group, 2)
        val_set.update(shuffled[:n_val])
        test_set.update(shuffled[n_val:])

    val_imgs, val_targets, val_lbls, val_pids, val_slice_idxs = [], [], [], [], []
    test_imgs, test_targets, test_lbls, test_pids, test_slice_idxs = [], [], [], [], []

    for p in sorted(val_set | test_set):
        p_dir = os.path.join(testing_dir, p)
        info = patient_info[p]
        group = info['Group']

        ed_idx = es_idx = None
        if restrict_to_systole:
            try:
                # ACDC Info.cfg uses 1-based frame indices; subtract 1 for 0-based numpy indexing
                ed_idx = int(info['ED']) - 1
                es_idx = int(info['ES']) - 1
            except (KeyError, ValueError) as e:
                print(f"Skipping ACDC {p}: missing ED/ES ({e})")
                continue
            if es_idx <= ed_idx:
                print(f"Skipping ACDC {p}: ES ({es_idx}) <= ED ({ed_idx})")
                continue

        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            if restrict_to_systole:
                imgs, targets, slcs = extract_consecutive_pairs_systole(
                    img_arr, 0, ed_idx, es_idx, target_size, return_slice_idxs=True)
                if not imgs:
                    print(f"Skipping ACDC {p}: ES ({es_idx}) out of range or no pairs")
            else:
                imgs, targets, slcs = extract_consecutive_pairs(img_arr, 0, target_size, return_slice_idxs=True)
            labels = [group] * len(imgs)
            pids = [p] * len(imgs)

            if p in val_set:
                val_imgs.extend(imgs); val_targets.extend(targets)
                val_lbls.extend(labels); val_pids.extend(pids)
                val_slice_idxs.extend(slcs)
            else:
                test_imgs.extend(imgs); test_targets.extend(targets)
                test_lbls.extend(labels); test_pids.extend(pids)
                test_slice_idxs.extend(slcs)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    return (_to_array(val_imgs), _to_array(val_targets), val_lbls, val_pids, val_slice_idxs,
            _to_array(test_imgs), _to_array(test_targets), test_lbls, test_pids, test_slice_idxs)


def load_mm_validation_data(mm_val_dir, csv_path, target_size=(128, 128), restrict_to_systole=False):
    df = pd.read_csv(csv_path)
    # subject_id -> (ed_idx, es_idx, pathology)
    subject_lookup = {row['External code']: (int(row['ED']), int(row['ES']), row['Pathology'])
                      for _, row in df.iterrows()}

    all_images, all_targets, all_labels, all_pids, all_slice_idxs = [], [], [], [], []
    sa_files = sorted([f for f in os.listdir(mm_val_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id in subject_lookup:
            ed_idx, es_idx, pathology = subject_lookup[subject_id]
        else:
            ed_idx, es_idx, pathology = None, None, 'UNKNOWN'

        if restrict_to_systole:
            if ed_idx is None:
                print(f"Skipping M&M {subject_id}: not in CSV (no ED/ES)")
                continue
            if es_idx <= ed_idx:
                print(f"Skipping M&M {subject_id}: ES ({es_idx}) <= ED ({ed_idx})")
                continue

        nii_path = os.path.join(mm_val_dir, fname)
        if not os.path.exists(nii_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            if restrict_to_systole:
                imgs, targets, slcs = extract_consecutive_pairs_systole(
                    img_arr, 0, ed_idx, es_idx, target_size, return_slice_idxs=True)
                if not imgs:
                    print(f"Skipping M&M {subject_id}: ES ({es_idx}) out of range or no pairs")
            else:
                imgs, targets, slcs = extract_consecutive_pairs(img_arr, 0, target_size, return_slice_idxs=True)

            all_images.extend(imgs)
            all_targets.extend(targets)
            all_labels.extend([pathology] * len(imgs))
            all_pids.extend([subject_id] * len(imgs))
            all_slice_idxs.extend(slcs)
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")

    return _to_array(all_images), _to_array(all_targets), all_labels, all_pids, all_slice_idxs


def load_acdc_test_val_ed_es_data(base_dir, target_size=(128, 128), seed=42):
    testing_dir = os.path.join(base_dir, 'database', 'testing')
    all_patient_dirs = sorted([d for d in os.listdir(testing_dir) if os.path.isdir(os.path.join(testing_dir, d))])
    
    group_patients, patient_info = {}, {}
    for p in all_patient_dirs:
        info = read_acdc_info(os.path.join(testing_dir, p, 'Info.cfg'))
        if group := info.get('Group'):
            patient_info[p] = info
            group_patients.setdefault(group, []).append(p)

    rng = random.Random(seed)
    val_set, test_set = set(), set()
    val_counts = {'NOR': 4, 'MINF': 2, 'DCM': 2, 'HCM': 2, 'RV': 2}

    for group, pats in sorted(group_patients.items()):
        shuffled = pats[:]
        rng.shuffle(shuffled)
        n_val = val_counts.get(group, 2)
        val_set.update(shuffled[:n_val])
        test_set.update(shuffled[n_val:])

    val_imgs, val_targets, val_lbls, val_pids, val_slice_idxs = [], [], [], [], []
    test_imgs, test_targets, test_lbls, test_pids, test_slice_idxs = [], [], [], [], []

    for p in sorted(val_set | test_set):
        p_dir = os.path.join(testing_dir, p)
        info = patient_info[p]
        if 'ED' not in info or 'ES' not in info: continue

        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        ed_frame_str = str(info['ED']).zfill(2)
        mask_path = os.path.join(p_dir, f'{p}_frame{ed_frame_str}_gt.nii.gz')

        if not os.path.exists(nii_path) or not os.path.exists(mask_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            # mask_arr = load_and_orient_sitk(mask_path)
            # FIXED: Pass mask_arr into the extractor!
            # imgs, targets = extract_edes_pairs(img_arr, mask_arr, int(info['ED']), int(info['ES']), target_size)
            # resize
            # ACDC Info.cfg uses 1-based frame indices; subtract 1 for 0-based numpy indexing
            imgs, targets, slcs = extract_edes_pairs(
                img_arr, 0, int(info['ED']) - 1, int(info['ES']) - 1, target_size, return_slice_idxs=True)
            labels, pids = [info['Group']] * len(imgs), [p] * len(imgs)

            if p in val_set:
                val_imgs.extend(imgs); val_targets.extend(targets)
                val_lbls.extend(labels); val_pids.extend(pids)
                val_slice_idxs.extend(slcs)
            else:
                test_imgs.extend(imgs); test_targets.extend(targets)
                test_lbls.extend(labels); test_pids.extend(pids)
                test_slice_idxs.extend(slcs)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    return (_to_array(val_imgs), _to_array(val_targets), val_lbls, val_pids, val_slice_idxs,
            _to_array(test_imgs), _to_array(test_targets), test_lbls, test_pids, test_slice_idxs)


def load_mm_validation_ed_es_data(mm_val_dir, csv_path, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    subject_lookup = {row['External code']: (int(row['ED']), int(row['ES']), row['Pathology']) for _, row in df.iterrows()}

    all_images, all_targets, all_labels, all_pids, all_slice_idxs = [], [], [], [], []
    sa_files = sorted([f for f in os.listdir(mm_val_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in subject_lookup: continue

        ed_idx, es_idx, pathology = subject_lookup[subject_id]
        nii_path = os.path.join(mm_val_dir, fname)
        mask_path = os.path.join(mm_val_dir, fname.replace('_sa.nii.gz', '_sa_gt.nii.gz'))

        if not os.path.exists(nii_path) or not os.path.exists(mask_path):
            continue

        try:
            img_arr = load_and_orient_sitk(nii_path)
            mask_arr = load_and_orient_sitk(mask_path)
            # FIXED: Pass mask_arr into the extractor!
            # imgs, targets = extract_edes_pairs(img_arr, mask_arr, ed_idx, es_idx, target_size)
            # resize
            imgs, targets, slcs = extract_edes_pairs(img_arr, 0, ed_idx, es_idx, target_size, return_slice_idxs=True)

            all_images.extend(imgs)
            all_targets.extend(targets)
            all_labels.extend([pathology] * len(imgs))
            all_pids.extend([subject_id] * len(imgs))
            all_slice_idxs.extend(slcs)
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")

    return _to_array(all_images), _to_array(all_targets), all_labels, all_pids, all_slice_idxs


def load_mm_testing_ed_es_data(mm_test_dir, csv_path, target_size=(128, 128)):
    """Load the M&M *Testing* set (ED/ES, all pathologies).

    The Testing folder shares the same file layout and CSV schema as
    Validation, so this is a thin wrapper around the validation loader.
    """
    return load_mm_validation_ed_es_data(mm_test_dir, csv_path, target_size)


def load_acdc_nor_training_with_pids(acdc_dir, target_size=(128, 128)):
    """Load ACDC NOR training subjects (ED/ES pairs) with per-sample patient IDs.

    Same data as load_acdc_ed_es_data but additionally returns a list of
    per-sample patient IDs aligned with the returned arrays — useful for
    per-patient or per-vendor analyses on the training distribution.
    """
    training_dir = os.path.join(acdc_dir, 'database', 'training')
    if not os.path.isdir(training_dir):
        training_dir = os.path.join(acdc_dir, 'database', 'training_test')

    patients = sorted([d for d in os.listdir(training_dir)
                       if os.path.isdir(os.path.join(training_dir, d))])
    images, targets, pids = [], [], []

    for p in patients:
        p_dir = os.path.join(training_dir, p)
        info  = read_acdc_info(os.path.join(p_dir, 'Info.cfg'))
        if info.get('Group', '') != 'NOR' or 'ED' not in info or 'ES' not in info:
            continue
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            continue
        try:
            img_arr = load_and_orient_sitk(nii_path)
            imgs, tgts = extract_edes_pairs(
                img_arr, 0, int(info['ED']) - 1, int(info['ES']) - 1, target_size)
            images.extend(imgs)
            targets.extend(tgts)
            pids.extend([p] * len(imgs))
        except Exception as e:
            print(f"Error processing ACDC {p}: {e}")

    print(f"ACDC NOR ED/ES samples: {len(images)} from {len(set(pids))} patients")
    return _to_array(images), _to_array(targets), pids


def load_mm_nor_training_with_pids(mm_training_dir, csv_path, target_size=(128, 128)):
    """Load M&M NOR training subjects (ED/ES pairs) with per-sample patient IDs."""
    df = pd.read_csv(csv_path)
    nor_rows = df[df['Pathology'] == 'NOR'][['External code', 'ED', 'ES']]
    nor_info = {row['External code']: (int(row['ED']), int(row['ES']))
                for _, row in nor_rows.iterrows()}

    images, targets, pids = [], [], []
    sa_files = sorted([f for f in os.listdir(mm_training_dir)
                       if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        sid = fname.replace('_sa.nii.gz', '')
        if sid not in nor_info:
            continue
        nii_path = os.path.join(mm_training_dir, fname)
        if not os.path.exists(nii_path):
            continue
        try:
            ed_idx, es_idx = nor_info[sid]
            img_arr = load_and_orient_sitk(nii_path)
            imgs, tgts = extract_edes_pairs(img_arr, 0, ed_idx, es_idx, target_size)
            images.extend(imgs)
            targets.extend(tgts)
            pids.extend([sid] * len(imgs))
        except Exception as e:
            print(f"Error processing M&M {sid}: {e}")

    print(f"M&M NOR ED/ES samples: {len(images)} from {len(set(pids))} patients")
    return _to_array(images), _to_array(targets), pids


def load_reconstructed_sax_data_rgb(recon_root, ed_es_csv, target_size=(128, 128), motion_threshold=0.001):
    """
    Load ED/ES frame pairs from P{nnn}_sax_recon.npy files for the RGB (ED-prediction) pipeline.
    All subjects are NOR — no pathology filtering needed.

    Parameters
    ----------
    recon_root : str   Directory containing P*_sax_recon.npy files.
    ed_es_csv  : str   Path to segmentation/ed_es_frames.csv
                       (columns: case_id, ed_frame, es_frame).
    target_size : tuple  (H, W), default (128, 128).
    motion_threshold : float  Minimum mean pixel difference to keep a slice (default 0.01).

    Returns
    -------
    all_es_images : np.ndarray  shape (N, H, W, 3)  – ES frames (model input)
    all_ed_images : np.ndarray  shape (N, H, W, 3)  – ED frames (reconstruction target)
    """
    df = pd.read_csv(ed_es_csv)
    ed_es_map = {row['case_id']: (int(row['ed_frame']), int(row['es_frame']))
                 for _, row in df.iterrows()}

    all_es_images, all_ed_images = [], []
    npy_files = sorted(
        f for f in os.listdir(recon_root) if f.endswith('_sax_recon.npy')
    )
    print(f"Reconstructed SAX: found {len(npy_files)} .npy files.")

    for fname in npy_files:
        case_id = fname.replace('_sax_recon.npy', '')
        if case_id not in ed_es_map:
            print(f"Skipping {case_id}: not in ED/ES CSV.")
            continue
        ed_idx, es_idx = ed_es_map[case_id]
        if ed_idx == es_idx:
            print(f"Skipping {case_id}: ED == ES ({ed_idx}).")
            continue

        cine = np.load(os.path.join(recon_root, fname))   # (T, Z, H, W)
        if cine.ndim != 4:
            print(f"Skipping {case_id}: unexpected shape {cine.shape}.")
            continue
        cine = _maybe_normalize_volume(cine, case_id)
        cine = _maybe_resample_volume(cine, _spacing_recon_default())
        T, Z, _, _ = cine.shape
        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {case_id}: ED={ed_idx}/ES={es_idx} out of range T={T}.")
            continue

        print(f"Recon {case_id}  ED={ed_idx}  ES={es_idx}")
        for z in _middle_slice_range(Z):
            slice_seq = cine[:, z, :, :].astype(np.float32)   # (T, H, W)
            h, w = slice_seq.shape[1], slice_seq.shape[2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            slice_seq = slice_seq[:, y0:y0 + side, x0:x0 + side]  # center-square crop

            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, 0, 0, target_size)
            ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, 0, 0, target_size)
            input_rgb, target_rgb = _edes_input_target(ed_rgb, es_rgb)

            if np.mean(np.abs(ed_rgb - es_rgb)) >= motion_threshold:
                all_es_images.append(input_rgb)
                all_ed_images.append(target_rgb)

    print(f"Reconstructed SAX RGB samples: {len(all_es_images)}")
    return _to_array(all_es_images), _to_array(all_ed_images)


# ── Standalone per-dataset ED/ES training loaders (RGB) ─────────────────────

def load_acdc_ed_es_data(acdc_dir, target_size=(128, 128)):
    """Load only ED/ES frame pairs from ACDC NOR training subjects (RGB pipeline)."""
    training_dir = os.path.join(acdc_dir, 'database', 'training')
    if not os.path.isdir(training_dir):
        training_dir = os.path.join(acdc_dir, 'database', 'training_test')

    patients = sorted([d for d in os.listdir(training_dir)
                       if os.path.isdir(os.path.join(training_dir, d))])
    all_images, all_targets = [], []

    for p in patients:
        p_dir = os.path.join(training_dir, p)
        info  = read_acdc_info(os.path.join(p_dir, 'Info.cfg'))
        if info.get('Group', '') != 'NOR' or 'ED' not in info or 'ES' not in info:
            continue
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            continue
        try:
            img_arr = load_and_orient_sitk(nii_path)
            # ACDC Info.cfg uses 1-based frame indices; subtract 1 for 0-based numpy indexing
            imgs, targets = extract_edes_pairs(
                img_arr, 0, int(info['ED']) - 1, int(info['ES']) - 1, target_size)
            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error processing ACDC {p}: {e}")

    print(f"ACDC NOR ED/ES samples: {len(all_images)}")
    return _to_array(all_images), _to_array(all_targets)


def load_mm_ed_es_data(mm_training_dir, csv_path, target_size=(128, 128)):
    """Load only ED/ES frame pairs from M&M NOR training subjects (RGB pipeline)."""
    df = pd.read_csv(csv_path)
    nor_rows = df[df['Pathology'] == 'NOR'][['External code', 'ED', 'ES']]
    nor_info = {row['External code']: (int(row['ED']), int(row['ES']))
                for _, row in nor_rows.iterrows()}
    print(f"M&M: found {len(nor_info)} NOR subjects in CSV.")

    all_images, all_targets = [], []
    sa_files = sorted([f for f in os.listdir(mm_training_dir)
                       if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in nor_info:
            continue
        nii_path = os.path.join(mm_training_dir, fname)
        if not os.path.exists(nii_path):
            continue
        try:
            ed_idx, es_idx = nor_info[subject_id]
            img_arr = load_and_orient_sitk(nii_path)
            imgs, targets = extract_edes_pairs(img_arr, 0, ed_idx, es_idx, target_size)
            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error processing M&M {subject_id}: {e}")

    print(f"M&M NOR ED/ES samples: {len(all_images)}")
    return _to_array(all_images), _to_array(all_targets)


# ── Reconstructed SAX next-frame loader (RGB) ────────────────────────────────

def load_reconstructed_sax_data_next_frame_rgb(recon_root, target_size=(128, 128),
                                                 restrict_to_systole=False, ed_es_csv=None):
    """Load all consecutive frame pairs from reconstructed SAX volumes (RGB pipeline).

    Unlike load_reconstructed_sax_data_rgb (ED/ES only), this iterates every
    adjacent pair t → t+1.  No ED/ES CSV is required by default.

    Parameters
    ----------
    recon_root  : str    Directory containing *_sax_recon.npy files.
    target_size : tuple  (H, W) resize target, default (128, 128).
    restrict_to_systole : bool
        If True, emit only pairs (t, t+1) for t ∈ [ed_idx, es_idx − 1]
        (systolic contraction).  Cases missing from ed_es_csv or with es ≤ ed
        are skipped.
    ed_es_csv : str | None
        Path to segmentation/ed_es_frames.csv.  Required when
        restrict_to_systole=True; ignored otherwise.

    Returns
    -------
    all_images  : np.ndarray  shape (N, H, W, 3)  – frame t   (model input)
    all_targets : np.ndarray  shape (N, H, W, 3)  – frame t+1 (reconstruction target)
    """
    ed_es_map = {}
    if restrict_to_systole:
        if ed_es_csv is None:
            raise ValueError(
                "load_reconstructed_sax_data_next_frame_rgb: ed_es_csv is required "
                "when restrict_to_systole=True"
            )
        df = pd.read_csv(ed_es_csv)
        ed_es_map = {row['case_id']: (int(row['ed_frame']), int(row['es_frame']))
                     for _, row in df.iterrows()}

    all_images, all_targets = [], []
    npy_files = sorted(f for f in os.listdir(recon_root) if f.endswith('_sax_recon.npy'))
    print(f"Reconstructed SAX (next-frame): found {len(npy_files)} .npy files.")

    for fname in npy_files:
        case_id = fname.replace('_sax_recon.npy', '')

        ed_idx = es_idx = None
        if restrict_to_systole:
            if case_id not in ed_es_map:
                print(f"Skipping Recon {case_id}: not in ED/ES CSV.")
                continue
            ed_idx, es_idx = ed_es_map[case_id]
            if es_idx <= ed_idx:
                print(f"Skipping Recon {case_id}: ES ({es_idx}) <= ED ({ed_idx})")
                continue

        cine = np.load(os.path.join(recon_root, fname))  # (T, Z, H, W)
        if cine.ndim != 4:
            print(f"Skipping {case_id}: unexpected shape {cine.shape}.")
            continue
        cine = _maybe_normalize_volume(cine, case_id)
        cine = _maybe_resample_volume(cine, _spacing_recon_default())
        T, Z, h, w = cine.shape

        # Centre-square crop to match the ED/ES loader behaviour
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        cine_cropped = cine[:, :, y0:y0 + side, x0:x0 + side].astype(np.float32)

        print(f"Recon {case_id}  T={T}  Z={Z}")
        if restrict_to_systole:
            imgs, targets = extract_consecutive_pairs_systole(
                cine_cropped, 0, ed_idx, es_idx, target_size)
            if not imgs:
                print(f"Skipping Recon {case_id}: ES ({es_idx}) out of range T={T} or no pairs")
        else:
            imgs, targets = extract_consecutive_pairs(cine_cropped, 0, target_size)
        all_images.extend(imgs)
        all_targets.extend(targets)

    print(f"Reconstructed SAX (next-frame) RGB samples: {len(all_images)}")
    return _to_array(all_images), _to_array(all_targets)