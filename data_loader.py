import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

def center_crop_or_pad(image, target_h=128, target_w=128):
    """Center crops an image, padding with zeros if it's smaller than the target size."""
    h, w = image.shape
    
    # 1. Pad if the image is smaller than the target size
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    if pad_h > 0 or pad_w > 0:
        # Pad evenly on both sides
        image = np.pad(
            image, 
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), 
            mode='constant', 
            constant_values=0
        )
        h, w = image.shape # Update dimensions after padding

    # 2. Calculate the center crop coordinates
    center_y, center_x = h // 2, w // 2
    start_y = center_y - target_h // 2
    start_x = center_x - target_w // 2
    
    # 3. Crop and return
    return image[start_y:start_y + target_h, start_x:start_x + target_w]
def aspect_preserve_resize(image, target_h=128, target_w=128):
    """Resizes image to fit within target dimensions while preserving aspect ratio, then pads."""
    h, w = image.shape
    
    # 1. Calculate the scale factor to fit the image inside the target box
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 2. Resize keeping the aspect ratio (no squashing)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 3. Calculate padding needed to reach 128x128
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    # 4. Pad with black pixels
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
def load_acdc_data(base_dir, target_size=(128, 128)):
    training_dir = os.path.join(base_dir, 'database', 'training')
    patients = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])

    all_images = []
    all_flows = []

    print(f"Found {len(patients)} patients.")

    for p in patients:
        p_dir = os.path.join(training_dir, p)
        cfg_path = os.path.join(p_dir, 'Info.cfg')
        if not os.path.exists(cfg_path):
            continue
            
        # Check group
        is_nor = False
        try:
            with open(cfg_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('Group'):
                        if 'NOR' in line:
                            is_nor = True
                        break
        except Exception as e:
            print(f"Error reading config for {p}: {e}")
            continue
        
        if not is_nor:
            continue
            
        print(f"Processing {p} (NOR)...")
        # Load 4D
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
             continue
             
        try:
            img_obj = sitk.ReadImage(nii_path)
            img_arr = sitk.GetArrayFromImage(img_obj) # (T, Z, Y, X) or similar. Normally ITK is (x,y,z,t) -> numpy (t,z,y,x)
        except Exception as e:
            print(f"Error loading image for {p}: {e}")
            continue

        if len(img_arr.shape) == 4:
            T, Z, H, W = img_arr.shape
        else:
             # Skip if not 4D
             print(f"Skipping {p}: Shape {img_arr.shape} is not 4D")
             continue
        
        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # Shape: (T, H, W)
             
            # --- APPLIED FIX: Robust 1st/99th Percentile Normalization ---
            # 1. Calculate the 1st and 99th percentiles across the whole sequence
            p1 = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)
            
            # 2. Calculate the range (adding 1e-8 to prevent division by zero)
            slice_range = p99 - p1 + 1e-8

            # Pre-process frames for this slice
            processed_frames = []
            
            for t in range(T):
                frame = slice_seq[t]
            
                # 3. Resize the frame
                frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
                    
                # 4. Normalize using the percentiles (NOT the absolute min/max)
                # If the slice is completely empty/black, p99-p1 will be ~0
                if (p99 - p1) < 1e-7:
                    frame_norm = np.zeros(target_size, dtype=np.float32)
                else:
                    frame_norm = (frame_cropped - p1) / slice_range
                    
                    # 5. Clip values strictly to [0.0, 1.0] 
                    # ANY outlier pixel brighter than p99 is now forced to exactly 1.0
                    # ANY outlier pixel darker than p1 is now forced to exactly 0.0
                    frame_norm = np.clip(frame_norm, 0.0, 1.0)

                frame_rgb = np.stack([frame_norm] * 3, axis=-1)
                processed_frames.append(frame_rgb)

            # ------------------------------
                    
            processed_frames_arr = np.array(processed_frames) # (T, 128, 128, 3)
             
            # Compute flows and pairs
            for t in range(T-1):
                prev_gray = (processed_frames_arr[t, :, :, 0] * 255).astype(np.uint8)
                next_gray = (processed_frames_arr[t+1, :, :, 0] * 255).astype(np.uint8)
                
                # Calc Dense Optical Flow
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                except Exception as e:
                    print(f"Flow failed: {e}")
                    continue

                # Add Magnitude channel
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # --- NEW: Check for dead/static slices ---
                # If the maximum pixel displacement is tiny, or the average displacement 
                # is virtually zero, skip this slice because the heart isn't beating here.
                if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                    # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                    continue
                # -----------------------------------------

                flow_3ch = np.dstack((flow, mag)) # (H, W, 3)
                
                all_images.append(processed_frames_arr[t]) 
                all_flows.append(flow_3ch)
                
    return np.array(all_images), np.array(all_flows)



def load_mm_data(mm_training_dir, csv_path, target_size=(128, 128)):
    """
    Load M&M (Multi-centre, Multi-vendor, Multi-disease) cardiac MRI data.
    Only subjects with Pathology == 'NOR' (normal) are loaded.

    Parameters
    ----------
    mm_training_dir : str
        Path to the M&M Training folder (Dataset_1/Training).
        Expected to contain files like {ID}_sa.nii.gz.
    csv_path : str
        Path to the M&M CSV metadata file
        (211230_M&Ms_Dataset_information_diagnosis_opendataset.csv).
    target_size : tuple
        (H, W) to which every frame is resized. Default matches load_acdc_data.

    Returns
    -------
    all_images : np.ndarray  shape (N, H, W, 3)
    all_flows  : np.ndarray  shape (N, H, W, 3)  [dx, dy, magnitude]
    """
    # Load CSV and build set of NOR subject IDs
    df = pd.read_csv(csv_path)
    nor_ids = set(df[df['Pathology'] == 'NOR']['External code'].tolist())
    print(f"Found {len(nor_ids)} M&M NOR subjects in CSV.")

    all_images = []
    all_flows  = []

    # Gather all short-axis volume files (skip ground-truth masks)
    sa_files = sorted([
        f for f in os.listdir(mm_training_dir)
        if f.endswith('_sa.nii.gz') and not f.endswith('_sa_gt.nii.gz')
    ])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in nor_ids:
            continue

        print(f"Processing {subject_id} (M&M NOR)...")
        nii_path = os.path.join(mm_training_dir, fname)

        try:
            img_obj = sitk.ReadImage(nii_path)
            img_arr = sitk.GetArrayFromImage(img_obj)  # typically (T, Z, H, W)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        if len(img_arr.shape) == 4:
            T, Z, H, W = img_arr.shape
        else:
            print(f"Skipping {subject_id}: unexpected shape {img_arr.shape}")
            continue

        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # (T, H, W)

            # --- APPLIED FIX: Robust 1st/99th Percentile Normalization ---
            p1 = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)
            slice_range = p99 - p1 + 1e-8

            processed_frames = []
            for t in range(T):
                frame = slice_seq[t]
                
                # Resize the frame first
                frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
                
                # Normalize using the percentiles
                if (p99 - p1) < 1e-7:
                    frame_norm = np.zeros(target_size, dtype=np.float32)
                else:
                    frame_norm = (frame_cropped - p1) / slice_range
                    # Clip values strictly to [0.0, 1.0] to crush extreme outliers
                    frame_norm = np.clip(frame_norm, 0.0, 1.0) 

                frame_rgb = np.stack([frame_norm] * 3, axis=-1)
                processed_frames.append(frame_rgb)

            processed_frames_arr = np.array(processed_frames)  # (T, H, W, 3)

            for t in range(T - 1):
                prev_gray = (processed_frames_arr[t,   :, :, 0] * 255).astype(np.uint8)
                next_gray = (processed_frames_arr[t+1, :, :, 0] * 255).astype(np.uint8)

                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                except Exception as e:
                    print(f"Flow failed for {subject_id} z={z} t={t}: {e}")
                    continue

                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # --- NEW: Check for dead/static slices ---
                # If the maximum pixel displacement is tiny, or the average displacement 
                # is virtually zero, skip this slice because the heart isn't beating here.
                if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                    # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                    continue
                # -----------------------------------------

                flow_3ch = np.dstack((flow, mag))  # (H, W, 3)

                all_images.append(processed_frames_arr[t])
                all_flows.append(flow_3ch)

    return np.array(all_images), np.array(all_flows)


def load_combined_data(acdc_dir, mm_training_dir, csv_path, target_size=(128, 128)):
    """
    Load and concatenate NOR patients from both ACDC (Dataset_2) and
    M&M (Dataset_1) datasets.

    Parameters
    ----------
    acdc_dir       : str    root of Dataset_2 (contains 'database/training')
    mm_training_dir: str    Dataset_1/Training folder
    csv_path       : str    M&M CSV metadata file path
    target_size    : tuple  (H, W) resize target, default (128, 128)

    Returns
    -------
    images : np.ndarray  shape (N_acdc + N_mm, H, W, 3)
    flows  : np.ndarray  shape (N_acdc + N_mm, H, W, 3)
    """
    print("=== Loading ACDC NOR data ===")
    acdc_images, acdc_flows = load_acdc_data(acdc_dir, target_size)
    print(f"ACDC NOR samples: {len(acdc_images)}")

    print("\n=== Loading M&M NOR data ===")
    mm_images, mm_flows = load_mm_data(mm_training_dir, csv_path, target_size)
    print(f"M&M NOR samples:  {len(mm_images)}")

    if len(acdc_images) == 0 and len(mm_images) == 0:
        return np.array([]), np.array([])
    elif len(acdc_images) == 0:
        return mm_images, mm_flows
    elif len(mm_images) == 0:
        return acdc_images, acdc_flows

    images = np.concatenate([acdc_images, mm_images], axis=0)
    flows  = np.concatenate([acdc_flows,  mm_flows],  axis=0)
    print(f"\nCombined: {len(images)} samples total")
    return images, flows

# ── ED/ES-only combined data loader ──────────────────────────────────────────
# Loads NOR subjects from ACDC (Dataset_2) and M&M (Dataset_1).
# For each subject and each z-slice:
#   • input image  = End-Systole (ES) frame
#   • next frame   = End-Diastole (ED) frame
# Optical flow is computed from ES → ED.

def load_combined_ed_es_data(acdc_dir, mm_training_dir, csv_path,
                              target_size=(128, 128)):
    """
    Load only End-Diastole (ED) and End-Systole (ES) frame pairs
    from both ACDC (Dataset_2) and M&M (Dataset_1) NOR subjects.

    Parameters
    ----------
    acdc_dir        : str   Root of Dataset_2 (contains 'database/training').
    mm_training_dir : str   Dataset_1/Training folder.
    csv_path        : str   M&M CSV metadata file path.
    target_size     : tuple (H, W) resize target, default (128, 128).

    Returns
    -------
    all_images : np.ndarray  shape (N, H, W, 3)  – ES frames (input)
    all_flows  : np.ndarray  shape (N, H, W, 3)  – optical flow ES→ED
    """

    def _preprocess_frame(frame, p1, p99, target_size):
        """Resize + percentile-normalise a single 2-D frame to RGB."""
        frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
        if (p99 - p1) < 1e-7:
            frame_norm = np.zeros(target_size, dtype=np.float32)
        else:
            frame_norm = np.clip((frame_cropped - p1) / (p99 - p1 + 1e-8),
                                 0.0, 1.0)
        return np.stack([frame_norm] * 3, axis=-1)  # (H, W, 3)

    all_images = []
    all_flows  = []

    # ── ACDC ────────────────────────────────────────────────────────────────
    training_dir = os.path.join(acdc_dir, 'database', 'training')
    if not os.path.isdir(training_dir):
        # fallback used by earlier experiments
        training_dir = os.path.join(acdc_dir, 'database', 'training_test')
    patients = sorted([d for d in os.listdir(training_dir)
                       if os.path.isdir(os.path.join(training_dir, d))])
    print(f"ACDC: found {len(patients)} patient folders.")

    for p in patients:
        p_dir    = os.path.join(training_dir, p)
        cfg_path = os.path.join(p_dir, 'Info.cfg')
        if not os.path.exists(cfg_path):
            continue

        # Read Group, ED, ES from Info.cfg
        info = {}
        try:
            with open(cfg_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, val = line.split(':', 1)
                        info[key.strip()] = val.strip()
        except Exception as e:
            print(f"Cannot read config for {p}: {e}")
            continue

        if info.get('Group', '') != 'NOR':
            continue

        try:
            ed_idx = int(info['ED']) 
            es_idx = int(info['ES'])
        except (KeyError, ValueError) as e:
            print(f"Missing ED/ES in {p}: {e}")
            continue
        if ed_idx == es_idx:
            print(f"Skipping ACDC {p}: ED and ES frames are identical ({ed_idx})")
            continue
        
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            continue

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))  # (T,Z,H,W)
        except Exception as e:
            print(f"Load error {p}: {e}")
            continue

        if img_arr.ndim != 4:
            continue
        T, Z, _, _ = img_arr.shape

        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {p}: ED={ed_idx} or ES={es_idx} out of range T={T}")
            continue

        print(f"ACDC {p} (NOR)  ED={ed_idx}  ES={es_idx}")
        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # (T, H, W)
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_frame_rgb = _preprocess_frame(slice_seq[es_idx], p1, p99, target_size)
            ed_frame_rgb = _preprocess_frame(slice_seq[ed_idx], p1, p99, target_size)

            es_gray = (es_frame_rgb[:, :, 0] * 255).astype(np.uint8)
            ed_gray = (ed_frame_rgb[:, :, 0] * 255).astype(np.uint8)

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    es_gray, ed_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception as e:
                print(f"Flow error {p} z={z}: {e}")
                continue

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # --- NEW: Check for dead/static slices ---
            # If the maximum pixel displacement is tiny, or the average displacement 
            # is virtually zero, skip this slice because the heart isn't beating here.
            if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                continue
            # -----------------------------------------

            flow_3ch = np.dstack((flow, mag))  # (H, W, 3)

            all_images.append(es_frame_rgb)
            all_flows.append(flow_3ch)

    print(f"ACDC NOR ED/ES samples: {len(all_images)}")
    acdc_count = len(all_images)

    # ── M&M ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    # Build lookup: subject_id -> (ed_idx, es_idx)
    nor_rows = df[df['Pathology'] == 'NOR'][['External code', 'ED', 'ES']]
    nor_info = {row['External code']: (int(row['ED']), int(row['ES']))
                for _, row in nor_rows.iterrows()}
    print(f"M&M: found {len(nor_info)} NOR subjects in CSV.")

    sa_files = sorted([f for f in os.listdir(mm_training_dir)
                       if f.endswith('_sa.nii.gz') and not f.endswith('_sa_gt.nii.gz')])

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in nor_info:
            continue

        ed_idx, es_idx = nor_info[subject_id]
        nii_path = os.path.join(mm_training_dir, fname)

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))  # (T,Z,H,W)
        except Exception as e:
            print(f"Load error {subject_id}: {e}")
            continue

        if img_arr.ndim != 4:
            continue
        T, Z, _, _ = img_arr.shape

        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {subject_id}: ED={ed_idx} or ES={es_idx} out of range T={T}")
            continue

        print(f"M&M {subject_id} (NOR)  ED={ed_idx}  ES={es_idx}")
        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # (T, H, W)
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_frame_rgb = _preprocess_frame(slice_seq[es_idx], p1, p99, target_size)
            ed_frame_rgb = _preprocess_frame(slice_seq[ed_idx], p1, p99, target_size)

            es_gray = (es_frame_rgb[:, :, 0] * 255).astype(np.uint8)
            ed_gray = (ed_frame_rgb[:, :, 0] * 255).astype(np.uint8)

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    es_gray, ed_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception as e:
                print(f"Flow error {subject_id} z={z}: {e}")
                continue

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # --- NEW: Check for dead/static slices ---
            # If the maximum pixel displacement is tiny, or the average displacement 
            # is virtually zero, skip this slice because the heart isn't beating here.
            if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                continue
            # -----------------------------------------

            flow_3ch = np.dstack((flow, mag))

            all_images.append(es_frame_rgb)
            all_flows.append(flow_3ch)

    mm_count = len(all_images) - acdc_count
    print(f"M&M NOR ED/ES samples: {mm_count}")
    print(f"Combined ED/ES samples: {len(all_images)}")

    if len(all_images) == 0:
        return np.array([]), np.array([])
    return np.array(all_images), np.array(all_flows)


# ── ACDC test-set loader with validation / test split ────────────────────────
#
# The ACDC *testing* set (database/testing) has 50 patients:
#   10 NOR  ·  10 MINF  ·  10 DCM  ·  10 HCM  ·  10 RV
#
# This loader randomly selects (with a fixed seed):
#   • Validation:  4 NOR  +  2 MINF  +  2 DCM  +  2 HCM  +  2 RV  = 12 patients
#   • Test:        6 NOR  +  8 MINF  +  8 DCM  +  8 HCM  +  8 RV  = 38 patients
#
# Each split is processed identically to load_acdc_data (consecutive frame
# pairs with optical flow).
# ─────────────────────────────────────────────────────────────────────────────

def load_acdc_test_val_data(base_dir, target_size=(128, 128), seed=42):
    """
    Load the ACDC *testing* set and split it into validation and test subsets.

    Split strategy (fixed random seed for reproducibility):
        Validation — 4 NOR + 2 per disease group (MINF, DCM, HCM, RV) = 12 patients
        Test       — remaining 6 NOR + 8 per disease group            = 38 patients

    Parameters
    ----------
    base_dir    : str    Root of Dataset_2 (contains 'database/testing').
    target_size : tuple  (H, W) resize target, default (128, 128).
    seed        : int    Random seed for reproducible patient selection.

    Returns
    -------
    val_images   : np.ndarray  shape (N_val, H, W, 3)
    val_flows    : np.ndarray  shape (N_val, H, W, 3)
    val_labels   : list[str]   per-sample disease group label
    val_pids     : list[str]   per-sample patient ID

    test_images  : np.ndarray  shape (N_test, H, W, 3)
    test_flows   : np.ndarray  shape (N_test, H, W, 3)
    test_labels  : list[str]   per-sample disease group label
    test_pids    : list[str]   per-sample patient ID
    """
    import random

    testing_dir = os.path.join(base_dir, 'database', 'testing')
    if not os.path.isdir(testing_dir):
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")

    # ── 1. Scan patients and group by disease ──────────────────────────────
    group_patients = {}  # e.g. {'NOR': ['patient102', ...], 'DCM': [...], ...}
    patient_info   = {}  # patient_id -> {'group': ..., 'ED': ..., 'ES': ...}

    all_patient_dirs = sorted([
        d for d in os.listdir(testing_dir)
        if os.path.isdir(os.path.join(testing_dir, d))
    ])

    for p in all_patient_dirs:
        cfg_path = os.path.join(testing_dir, p, 'Info.cfg')
        if not os.path.exists(cfg_path):
            continue

        info = {}
        try:
            with open(cfg_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, val = line.split(':', 1)
                        info[key.strip()] = val.strip()
        except Exception as e:
            print(f"Cannot read config for {p}: {e}")
            continue

        group = info.get('Group', '')
        if group == '':
            continue

        patient_info[p] = info
        group_patients.setdefault(group, []).append(p)

    print("ACDC testing set composition:")
    for g in sorted(group_patients):
        print(f"  {g}: {len(group_patients[g])} patients -> {group_patients[g]}")

    # ── 2. Split into val / test patient lists ─────────────────────────────
    rng = random.Random(seed)

    val_patient_set  = set()
    test_patient_set = set()

    # Number of patients to sample for validation per group
    val_counts = {'NOR': 4, 'MINF': 2, 'DCM': 2, 'HCM': 2, 'RV': 2}

    for group, patients_in_group in sorted(group_patients.items()):
        n_val = val_counts.get(group, 2)  # default 2 for any unexpected group
        shuffled = patients_in_group[:]
        rng.shuffle(shuffled)
        val_patients  = shuffled[:n_val]
        test_patients = shuffled[n_val:]
        val_patient_set.update(val_patients)
        test_patient_set.update(test_patients)

    print(f"\nValidation patients ({len(val_patient_set)}): "
          f"{sorted(val_patient_set)}")
    print(f"Test patients ({len(test_patient_set)}): "
          f"{sorted(test_patient_set)}")

    # ── 3. Processing helper (same logic as load_acdc_data) ────────────────
    def _process_patient(p, p_dir, group):
        """Return (images, flows, labels, pids) lists for one patient."""
        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            return [], [], [], []

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return [], [], [], []

        if img_arr.ndim != 4:
            print(f"Skipping {p}: shape {img_arr.shape} is not 4D")
            return [], [], [], []

        T, Z, H, W = img_arr.shape

        images, flows, labels, pids = [], [], [], []

        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # (T, H, W)
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)
            slice_range = p99 - p1 + 1e-8

            processed_frames = []
            for t in range(T):
                frame = slice_seq[t]
                frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
                if (p99 - p1) < 1e-7:
                    frame_norm = np.zeros(target_size, dtype=np.float32)
                else:
                    frame_norm = np.clip(
                        (frame_cropped - p1) / slice_range, 0.0, 1.0
                    )
                frame_rgb = np.stack([frame_norm] * 3, axis=-1)
                processed_frames.append(frame_rgb)

            processed_frames_arr = np.array(processed_frames)  # (T, H, W, 3)

            for t in range(T - 1):
                prev_gray = (processed_frames_arr[t,   :, :, 0] * 255).astype(np.uint8)
                next_gray = (processed_frames_arr[t+1, :, :, 0] * 255).astype(np.uint8)

                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                except Exception as e:
                    print(f"Flow failed {p} z={z} t={t}: {e}")
                    continue

                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # --- NEW: Check for dead/static slices ---
                # If the maximum pixel displacement is tiny, or the average displacement 
                # is virtually zero, skip this slice because the heart isn't beating here.
                if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                    # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                    continue
                # -----------------------------------------

                flow_3ch = np.dstack((flow, mag))

                images.append(processed_frames_arr[t])
                flows.append(flow_3ch)
                labels.append(group)
                pids.append(p)

        return images, flows, labels, pids

    # ── 4. Load and process each split ─────────────────────────────────────
    val_images,  val_flows,  val_labels,  val_pids  = [], [], [], []
    test_images, test_flows, test_labels, test_pids = [], [], [], []

    for p in sorted(val_patient_set | test_patient_set):
        p_dir = os.path.join(testing_dir, p)
        group = patient_info[p]['Group']
        print(f"Processing {p} ({group})...")

        imgs, fls, lbls, pds = _process_patient(p, p_dir, group)

        if p in val_patient_set:
            val_images.extend(imgs)
            val_flows.extend(fls)
            val_labels.extend(lbls)
            val_pids.extend(pds)
        else:
            test_images.extend(imgs)
            test_flows.extend(fls)
            test_labels.extend(lbls)
            test_pids.extend(pds)

    print(f"\n{'='*50}")
    print(f"Validation: {len(val_images)} samples from {len(val_patient_set)} patients")
    print(f"Test:       {len(test_images)} samples from {len(test_patient_set)} patients")

    def _to_array(lst):
        return np.array(lst) if len(lst) > 0 else np.array([])

    return (_to_array(val_images),  _to_array(val_flows),  val_labels,  val_pids,
            _to_array(test_images), _to_array(test_flows), test_labels, test_pids)


# ── M&M Validation loader (ALL pathologies) ─────────────────────────────────
#
# Loads every subject in the M&M *Validation* folder (Dataset_1/Validation),
# regardless of pathology.  Returns per-sample disease labels and patient IDs
# so downstream code can analyse each group separately.
# ─────────────────────────────────────────────────────────────────────────────

def load_mm_validation_data(mm_val_dir, csv_path, target_size=(128, 128)):
    """
    Load ALL subjects from the M&M Validation folder (every pathology).

    Parameters
    ----------
    mm_val_dir  : str    Path to Dataset_1/Validation.
    csv_path    : str    Path to the M&M CSV metadata file.
    target_size : tuple  (H, W) resize target, default (128, 128).

    Returns
    -------
    all_images : np.ndarray   shape (N, H, W, 3)
    all_flows  : np.ndarray   shape (N, H, W, 3)
    all_labels : list[str]    per-sample pathology label
    all_pids   : list[str]    per-sample subject ID
    """
    # Build lookup: subject_id -> pathology
    df = pd.read_csv(csv_path)
    pathology_map = dict(zip(df['External code'], df['Pathology']))

    all_images = []
    all_flows  = []
    all_labels = []
    all_pids   = []

    sa_files = sorted([
        f for f in os.listdir(mm_val_dir)
        if f.endswith('_sa.nii.gz') and not f.endswith('_sa_gt.nii.gz')
    ])
    print(f"M&M Validation: found {len(sa_files)} subjects.")

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        pathology  = pathology_map.get(subject_id, 'UNKNOWN')
        nii_path   = os.path.join(mm_val_dir, fname)

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
            continue

        if img_arr.ndim != 4:
            print(f"Skipping {subject_id}: unexpected shape {img_arr.shape}")
            continue

        T, Z, H, W = img_arr.shape
        print(f"Processing {subject_id} ({pathology})...")

        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]  # (T, H, W)
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)
            slice_range = p99 - p1 + 1e-8

            processed_frames = []
            for t in range(T):
                frame = slice_seq[t]
                frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
                if (p99 - p1) < 1e-7:
                    frame_norm = np.zeros(target_size, dtype=np.float32)
                else:
                    frame_norm = np.clip(
                        (frame_cropped - p1) / slice_range, 0.0, 1.0
                    )
                frame_rgb = np.stack([frame_norm] * 3, axis=-1)
                processed_frames.append(frame_rgb)

            processed_frames_arr = np.array(processed_frames)  # (T, H, W, 3)

            for t in range(T - 1):
                prev_gray = (processed_frames_arr[t,   :, :, 0] * 255).astype(np.uint8)
                next_gray = (processed_frames_arr[t+1, :, :, 0] * 255).astype(np.uint8)

                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                except Exception as e:
                    print(f"Flow failed {subject_id} z={z} t={t}: {e}")
                    continue

                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # --- NEW: Check for dead/static slices ---
                # If the maximum pixel displacement is tiny, or the average displacement 
                # is virtually zero, skip this slice because the heart isn't beating here.
                if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                    # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                    continue
                # -----------------------------------------

                flow_3ch = np.dstack((flow, mag))

                all_images.append(processed_frames_arr[t])
                all_flows.append(flow_3ch)
                all_labels.append(pathology)
                all_pids.append(subject_id)

    print(f"M&M Validation total: {len(all_images)} samples from {len(sa_files)} subjects")

    def _to_array(lst):
        return np.array(lst) if len(lst) > 0 else np.array([])

    return _to_array(all_images), _to_array(all_flows), all_labels, all_pids


# ── ACDC test-set loader: ED/ES only, with val / test split ─────────────────
#
# Same patient split as load_acdc_test_val_data, but loads ONLY the
# End-Diastole (ED) and End-Systole (ES) frame pair per z-slice.
# Flow is computed ES → ED (matching load_combined_ed_es_data).
# ─────────────────────────────────────────────────────────────────────────────

def load_acdc_test_val_ed_es_data(base_dir, target_size=(128, 128), seed=42):
    """
    Load the ACDC *testing* set using only ED/ES frames and split into
    validation and test subsets.

    Split strategy (fixed random seed):
        Validation — 4 NOR + 2 per disease group (MINF, DCM, HCM, RV) = 12 patients
        Test       — remaining 6 NOR + 8 per disease group            = 38 patients

    Returns
    -------
    val_images, val_flows, val_labels, val_pids,
    test_images, test_flows, test_labels, test_pids
    """
    import random

    def _preprocess_frame(frame, p1, p99, target_size):
        frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
        if (p99 - p1) < 1e-7:
            return np.stack([np.zeros(target_size, dtype=np.float32)] * 3, axis=-1)
        frame_norm = np.clip((frame_cropped - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        return np.stack([frame_norm] * 3, axis=-1)

    testing_dir = os.path.join(base_dir, 'database', 'testing')
    if not os.path.isdir(testing_dir):
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")

    # ── 1. Scan patients and group by disease ──────────────────────────────
    group_patients = {}
    patient_info   = {}

    for p in sorted(os.listdir(testing_dir)):
        p_dir = os.path.join(testing_dir, p)
        if not os.path.isdir(p_dir):
            continue
        cfg_path = os.path.join(p_dir, 'Info.cfg')
        if not os.path.exists(cfg_path):
            continue
        info = {}
        try:
            with open(cfg_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, val = line.split(':', 1)
                        info[key.strip()] = val.strip()
        except Exception as e:
            print(f"Cannot read config for {p}: {e}")
            continue
        group = info.get('Group', '')
        if group == '':
            continue
        patient_info[p] = info
        group_patients.setdefault(group, []).append(p)

    print("ACDC testing set composition:")
    for g in sorted(group_patients):
        print(f"  {g}: {len(group_patients[g])} patients")

    # ── 2. Split ───────────────────────────────────────────────────────────
    rng = random.Random(seed)
    val_patient_set, test_patient_set = set(), set()
    val_counts = {'NOR': 4, 'MINF': 2, 'DCM': 2, 'HCM': 2, 'RV': 2}

    for group, pats in sorted(group_patients.items()):
        n_val = val_counts.get(group, 2)
        shuffled = pats[:]
        rng.shuffle(shuffled)
        val_patient_set.update(shuffled[:n_val])
        test_patient_set.update(shuffled[n_val:])

    print(f"\nValidation patients ({len(val_patient_set)}): {sorted(val_patient_set)}")
    print(f"Test patients ({len(test_patient_set)}): {sorted(test_patient_set)}")

    # ── 3. Process each patient (ED/ES only) ───────────────────────────────
    def _process_patient_ed_es(p, p_dir, group):
        info = patient_info[p]
        try:
            ed_idx = int(info['ED'])
            es_idx = int(info['ES'])
        except (KeyError, ValueError) as e:
            print(f"Missing ED/ES in {p}: {e}")
            return [], [], [], []
        if ed_idx == es_idx:
            print(f"Skipping {p}: ED and ES identical ({ed_idx})")
            return [], [], [], []

        nii_path = os.path.join(p_dir, f'{p}_4d.nii.gz')
        if not os.path.exists(nii_path):
            return [], [], [], []

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return [], [], [], []
        if img_arr.ndim != 4:
            return [], [], [], []
        T, Z, _, _ = img_arr.shape
        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {p}: ED={ed_idx} or ES={es_idx} out of range T={T}")
            return [], [], [], []

        images, flows, labels, pids = [], [], [], []
        print(f"  {p} ({group})  ED={ed_idx}  ES={es_idx}")

        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_rgb = _preprocess_frame(slice_seq[es_idx], p1, p99, target_size)
            ed_rgb = _preprocess_frame(slice_seq[ed_idx], p1, p99, target_size)

            es_gray = (es_rgb[:, :, 0] * 255).astype(np.uint8)
            ed_gray = (ed_rgb[:, :, 0] * 255).astype(np.uint8)

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    es_gray, ed_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception as e:
                print(f"Flow error {p} z={z}: {e}")
                continue

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # --- NEW: Check for dead/static slices ---
            # If the maximum pixel displacement is tiny, or the average displacement 
            # is virtually zero, skip this slice because the heart isn't beating here.
            if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                continue
            # -----------------------------------------

            flow_3ch = np.dstack((flow, mag))

            images.append(es_rgb)
            flows.append(flow_3ch)
            labels.append(group)
            pids.append(p)

        return images, flows, labels, pids

    # ── 4. Load each split ─────────────────────────────────────────────────
    val_images,  val_flows,  val_labels,  val_pids  = [], [], [], []
    test_images, test_flows, test_labels, test_pids = [], [], [], []

    for p in sorted(val_patient_set | test_patient_set):
        p_dir = os.path.join(testing_dir, p)
        group = patient_info[p]['Group']
        imgs, fls, lbls, pds = _process_patient_ed_es(p, p_dir, group)

        if p in val_patient_set:
            val_images.extend(imgs);  val_flows.extend(fls)
            val_labels.extend(lbls);  val_pids.extend(pds)
        else:
            test_images.extend(imgs); test_flows.extend(fls)
            test_labels.extend(lbls); test_pids.extend(pds)

    print(f"\nValidation: {len(val_images)} ED/ES samples from {len(val_patient_set)} patients")
    print(f"Test:       {len(test_images)} ED/ES samples from {len(test_patient_set)} patients")

    def _to_array(lst):
        return np.array(lst) if len(lst) > 0 else np.array([])

    return (_to_array(val_images),  _to_array(val_flows),  val_labels,  val_pids,
            _to_array(test_images), _to_array(test_flows), test_labels, test_pids)


# ── M&M Validation loader: ED/ES only, ALL pathologies ──────────────────────
#
# Loads every subject in Dataset_1/Validation using only the ED and ES frames.
# Flow is computed ES → ED.
# ─────────────────────────────────────────────────────────────────────────────

def load_mm_validation_ed_es_data(mm_val_dir, csv_path, target_size=(128, 128)):
    """
    Load ALL subjects from M&M Validation folder using only ED/ES frames.

    Parameters
    ----------
    mm_val_dir  : str    Path to Dataset_1/Validation.
    csv_path    : str    Path to M&M CSV metadata file.
    target_size : tuple  (H, W) resize target, default (128, 128).

    Returns
    -------
    all_images : np.ndarray   shape (N, H, W, 3)   – ES frames
    all_flows  : np.ndarray   shape (N, H, W, 3)   – optical flow ES→ED
    all_labels : list[str]    per-sample pathology label
    all_pids   : list[str]    per-sample subject ID
    """
    def _preprocess_frame(frame, p1, p99, target_size):
        frame_cropped = aspect_preserve_resize(
                    frame.astype(np.float32), 
                    target_size[0], 
                    target_size[1]
                )
        if (p99 - p1) < 1e-7:
            return np.stack([np.zeros(target_size, dtype=np.float32)] * 3, axis=-1)
        frame_norm = np.clip((frame_cropped - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        return np.stack([frame_norm] * 3, axis=-1)

    df = pd.read_csv(csv_path)
    # Build lookup: subject_id -> (ed_idx, es_idx, pathology)
    subject_lookup = {}
    for _, row in df.iterrows():
        sid = row['External code']
        subject_lookup[sid] = (int(row['ED']), int(row['ES']), row['Pathology'])

    all_images, all_flows, all_labels, all_pids = [], [], [], []

    sa_files = sorted([
        f for f in os.listdir(mm_val_dir)
        if f.endswith('_sa.nii.gz') and not f.endswith('_sa_gt.nii.gz')
    ])
    print(f"M&M Validation (ED/ES): found {len(sa_files)} subjects.")

    for fname in sa_files:
        subject_id = fname.replace('_sa.nii.gz', '')
        if subject_id not in subject_lookup:
            print(f"Skipping {subject_id}: not found in CSV")
            continue

        ed_idx, es_idx, pathology = subject_lookup[subject_id]
        if ed_idx == es_idx:
            print(f"Skipping {subject_id}: ED and ES identical ({ed_idx})")
            continue

        nii_path = os.path.join(mm_val_dir, fname)
        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
            continue

        if img_arr.ndim != 4:
            print(f"Skipping {subject_id}: unexpected shape {img_arr.shape}")
            continue

        T, Z, _, _ = img_arr.shape
        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {subject_id}: ED={ed_idx} or ES={es_idx} out of range T={T}")
            continue

        print(f"  {subject_id} ({pathology})  ED={ed_idx}  ES={es_idx}")

        for z in range(Z):
            slice_seq = img_arr[:, z, :, :]
            p1  = np.percentile(slice_seq, 1)
            p99 = np.percentile(slice_seq, 99)

            es_rgb = _preprocess_frame(slice_seq[es_idx], p1, p99, target_size)
            ed_rgb = _preprocess_frame(slice_seq[ed_idx], p1, p99, target_size)

            es_gray = (es_rgb[:, :, 0] * 255).astype(np.uint8)
            ed_gray = (ed_rgb[:, :, 0] * 255).astype(np.uint8)

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    es_gray, ed_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception as e:
                print(f"Flow error {subject_id} z={z}: {e}")
                continue

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # --- NEW: Check for dead/static slices ---
            # If the maximum pixel displacement is tiny, or the average displacement 
            # is virtually zero, skip this slice because the heart isn't beating here.
            if np.mean(mag) < 0.05 or np.max(mag) < 0.5:
                # print(f"Skipping static slice z={z} (Mean Flow: {np.mean(mag):.3f})")
                continue
            # -----------------------------------------

            flow_3ch = np.dstack((flow, mag))

            all_images.append(es_rgb)
            all_flows.append(flow_3ch)
            all_labels.append(pathology)
            all_pids.append(subject_id)

    print(f"M&M Validation ED/ES total: {len(all_images)} samples from {len(sa_files)} subjects")

    def _to_array(lst):
        return np.array(lst) if len(lst) > 0 else np.array([])

    return _to_array(all_images), _to_array(all_flows), all_labels, all_pids
