import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import random

# =============================================================================
# 1. CORE IMAGE UTILITIES
# =============================================================================

def load_and_orient_sitk(nii_path):
    """Safely loads 4D NIfTI images WITHOUT breaking the oblique SAX plane."""
    image = sitk.ReadImage(str(nii_path))
    return sitk.GetArrayFromImage(image)

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

def extract_consecutive_pairs(img_arr, mask_arr, target_size=(128, 128), motion_threshold=0.01):
    """Extracts valid, moving (t, t+1) frame pairs from a 4D array."""
    if img_arr.ndim != 4:
        return [], []
        
    T, Z, _, _ = img_arr.shape
    
    # GET CENTERS FOR ALL Z-SLICES ONCE
    #centers = get_slice_centers_from_mask(mask_arr)
    images, target_frames = [], []

    for z in range(1, Z - 1):
        #cy, cx = centers[z] # Unpack the center
        slice_seq = img_arr[:, z, :, :]
        p1, p99 = np.percentile(slice_seq, 1), np.percentile(slice_seq, 99)
        
        # Pass cy and cx into the normalizer
        #processed = [normalize_frame_to_rgb(slice_seq[t], p1, p99, cy, cx, target_size) for t in range(T)]
        # Use aspect preserve resize
        # Pass 0, 0 for center_y and center_x
        processed = [normalize_frame_to_rgb(slice_seq[t], p1, p99, 0, 0, target_size) for t in range(T)]

        for t in range(T - 1):
            f1, f2 = processed[t], processed[t+1]
            if np.mean(np.abs(f2 - f1)) >= motion_threshold:
                images.append(f1)
                target_frames.append(f2)
                
    return images, target_frames

def extract_edes_pairs(img_arr,mask_arr, ed_idx, es_idx, target_size=(128, 128), motion_threshold=0.01):
    if img_arr.ndim != 4:
        return [], []
        
    T, Z, _, _ = img_arr.shape
    if max(ed_idx, es_idx) >= T or ed_idx == es_idx:
        return [], []
        
    # GET CENTERS FOR ALL Z-SLICES ONCE
    # centers = get_slice_centers_from_mask(mask_arr)
    images, target_frames = [], []

    for z in range(1, Z - 1):
        # cy, cx = centers[z] # Unpack the center for this specific slice

        slice_seq = img_arr[:, z, :, :]
        p1, p99 = np.percentile(slice_seq, 1), np.percentile(slice_seq, 99)
        
        # Pass cy and cx into the normalizer
        # es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, cy, cx, target_size)
        # ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, cy, cx, target_size)
        # resize
        es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, 0, 0, target_size)
        ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, 0, 0, target_size)
        
        if np.mean(np.abs(ed_rgb - es_rgb)) >= motion_threshold:
            images.append(es_rgb)
            target_frames.append(ed_rgb)
            
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

def load_acdc_data(base_dir, target_size=(128, 128)):
    training_dir = os.path.join(base_dir, 'database', 'training')
    patients = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])
    
    all_images, all_targets = [], []
    print(f"Found {len(patients)} patients in ACDC.")

    for p in patients:
        p_dir = os.path.join(training_dir, p)
        info = read_acdc_info(os.path.join(p_dir, 'Info.cfg'))
        
        if info.get('Group', '') != 'NOR' or 'ED' not in info:
            continue
            
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        # Find the ED mask filename to use as our center reference
        ed_frame_str = str(info['ED']).zfill(2)
        mask_path = os.path.join(p_dir, f'{p}_frame{ed_frame_str}_gt.nii.gz')
        
        if not os.path.exists(nii_path) or not os.path.exists(mask_path): 
            continue
            
        try:
            img_arr = load_and_orient_sitk(nii_path)
            # mask_arr = load_and_orient_sitk(mask_path) # Shape: (Z, H, W)
            
            # FIXED: Call consecutive pairs extractor, not ED/ES!
            # imgs, targets = extract_consecutive_pairs(img_arr, mask_arr, target_size)
            # resize
            imgs, targets = extract_consecutive_pairs(img_arr, 0, target_size)

            all_images.extend(imgs)
            all_targets.extend(targets)
        except Exception as e:
            print(f"Error loading {p}: {e}")

    return _to_array(all_images), _to_array(all_targets)


def load_mm_data(mm_training_dir, csv_path, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    nor_ids = set(df[df['Pathology'] == 'NOR']['External code'].tolist())
    print(f"Found {len(nor_ids)} M&M NOR subjects in CSV.")

    all_images, all_targets = [], []
    sa_files = sorted([f for f in os.listdir(mm_training_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        # FIXED: Use nor_ids, not nor_info!
        if subject_id not in nor_ids: 
            continue
        
        nii_path = os.path.join(mm_training_dir, fname)
        mask_path = os.path.join(mm_training_dir, fname.replace('_sa.nii.gz', '_sa_gt.nii.gz'))
        
        if not os.path.exists(mask_path) or not os.path.exists(nii_path):
            continue
            
        try:
            img_arr = load_and_orient_sitk(nii_path)
            mask_arr = load_and_orient_sitk(mask_path) # Shape: (T, Z, H, W)
            
            # FIXED: Call consecutive pairs extractor!
            # imgs, targets = extract_consecutive_pairs(img_arr, mask_arr, target_size)
            # resize
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
            imgs, targets = extract_edes_pairs(img_arr, 0, int(info['ED']), int(info['ES']), target_size)
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


def load_acdc_test_val_data(base_dir, target_size=(128, 128), seed=42):
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

    val_imgs, val_targets, val_lbls, val_pids = [], [], [], []
    test_imgs, test_targets, test_lbls, test_pids = [], [], [], []

    for p in sorted(val_set | test_set):
        p_dir = os.path.join(testing_dir, p)
        info = patient_info[p]
        group = info['Group']
        
        if 'ED' not in info:
            continue
            
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        ed_frame_str = str(info['ED']).zfill(2)
        mask_path = os.path.join(p_dir, f'{p}_frame{ed_frame_str}_gt.nii.gz')
        
        if not os.path.exists(nii_path) or not os.path.exists(mask_path): 
            continue
            
        try:
            img_arr = load_and_orient_sitk(nii_path)
            # mask_arr = load_and_orient_sitk(mask_path)
            
            # FIXED: Call consecutive pairs extractor!
            # imgs, targets = extract_consecutive_pairs(img_arr, mask_arr, target_size)
            # resize
            imgs, targets = extract_consecutive_pairs(img_arr, 0, target_size)
            labels = [group] * len(imgs)
            pids = [p] * len(imgs)
            
            if p in val_set:
                val_imgs.extend(imgs); val_targets.extend(targets)
                val_lbls.extend(labels); val_pids.extend(pids)
            else:
                test_imgs.extend(imgs); test_targets.extend(targets)
                test_lbls.extend(labels); test_pids.extend(pids)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    return (_to_array(val_imgs), _to_array(val_targets), val_lbls, val_pids,
            _to_array(test_imgs), _to_array(test_targets), test_lbls, test_pids)


def load_mm_validation_data(mm_val_dir, csv_path, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    pathology_map = dict(zip(df['External code'], df['Pathology']))
    
    all_images, all_targets, all_labels, all_pids = [], [], [], []
    sa_files = sorted([f for f in os.listdir(mm_val_dir) if f.endswith('_sa.nii.gz') and not f.endswith('_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        pathology = pathology_map.get(subject_id, 'UNKNOWN')
        
        nii_path = os.path.join(mm_val_dir, fname)
        mask_path = os.path.join(mm_val_dir, fname.replace('_sa.nii.gz', '_sa_gt.nii.gz'))
        
        if not os.path.exists(nii_path) or not os.path.exists(mask_path):
            continue
            
        try:
            img_arr = load_and_orient_sitk(nii_path)
            # mask_arr = load_and_orient_sitk(mask_path) 
            
            # FIXED: Call consecutive pairs extractor!
            # imgs, targets = extract_consecutive_pairs(img_arr, mask_arr, target_size)
            # resize
            imgs, targets = extract_consecutive_pairs(img_arr, 0, target_size)
            
            all_images.extend(imgs)
            all_targets.extend(targets)
            all_labels.extend([pathology] * len(imgs))
            all_pids.extend([subject_id] * len(imgs))
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")

    return _to_array(all_images), _to_array(all_targets), all_labels, all_pids


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

    val_imgs, val_targets, val_lbls, val_pids = [], [], [], []
    test_imgs, test_targets, test_lbls, test_pids = [], [], [], []

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
            imgs, targets = extract_edes_pairs(img_arr, 0, int(info['ED']), int(info['ES']), target_size)
            labels, pids = [info['Group']] * len(imgs), [p] * len(imgs)
            
            if p in val_set:
                val_imgs.extend(imgs); val_targets.extend(targets)
                val_lbls.extend(labels); val_pids.extend(pids)
            else:
                test_imgs.extend(imgs); test_targets.extend(targets)
                test_lbls.extend(labels); test_pids.extend(pids)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    return (_to_array(val_imgs), _to_array(val_targets), val_lbls, val_pids,
            _to_array(test_imgs), _to_array(test_targets), test_lbls, test_pids)


def load_mm_validation_ed_es_data(mm_val_dir, csv_path, target_size=(128, 128)):
    df = pd.read_csv(csv_path)
    subject_lookup = {row['External code']: (int(row['ED']), int(row['ES']), row['Pathology']) for _, row in df.iterrows()}
    
    all_images, all_targets, all_labels, all_pids = [], [], [], []
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
            imgs, targets = extract_edes_pairs(img_arr, 0, ed_idx, es_idx, target_size)
            
            all_images.extend(imgs)
            all_targets.extend(targets)
            all_labels.extend([pathology] * len(imgs))
            all_pids.extend([subject_id] * len(imgs))
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")

    return _to_array(all_images), _to_array(all_targets), all_labels, all_pids


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
        T, Z, _, _ = cine.shape
        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {case_id}: ED={ed_idx}/ES={es_idx} out of range T={T}.")
            continue

        print(f"Recon {case_id}  ED={ed_idx}  ES={es_idx}")
        for z in range(1, Z - 1):
            slice_seq = cine[:, z, :, :].astype(np.float32)   # (T, H, W)
            h, w = slice_seq.shape[1], slice_seq.shape[2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            slice_seq = slice_seq[:, y0:y0 + side, x0:x0 + side]  # center-square crop

            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_rgb = normalize_frame_to_rgb(slice_seq[es_idx], p1, p99, 0, 0, target_size)
            ed_rgb = normalize_frame_to_rgb(slice_seq[ed_idx], p1, p99, 0, 0, target_size)

            if np.mean(np.abs(ed_rgb - es_rgb)) >= motion_threshold:
                all_es_images.append(es_rgb)
                all_ed_images.append(ed_rgb)

    print(f"Reconstructed SAX RGB samples: {len(all_es_images)}")
    return _to_array(all_es_images), _to_array(all_ed_images)