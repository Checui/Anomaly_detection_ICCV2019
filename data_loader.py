def load_acdc_data(base_dir, target_size=(128, 192)):
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
                frame_resized = cv2.resize(
                    frame.astype(np.float32),
                    (target_size[1], target_size[0])
                )
                    
                # 4. Normalize using the percentiles (NOT the absolute min/max)
                # If the slice is completely empty/black, p99-p1 will be ~0
                if (p99 - p1) < 1e-7:
                    frame_norm = np.zeros(target_size, dtype=np.float32)
                else:
                    frame_norm = (frame_resized - p1) / slice_range
                    
                    # 5. Clip values strictly to [0.0, 1.0] 
                    # ANY outlier pixel brighter than p99 is now forced to exactly 1.0
                    # ANY outlier pixel darker than p1 is now forced to exactly 0.0
                    frame_norm = np.clip(frame_norm, 0.0, 1.0)

                frame_rgb = np.stack([frame_norm] * 3, axis=-1)
                processed_frames.append(frame_rgb)

            # ------------------------------
                    
            processed_frames_arr = np.array(processed_frames) # (T, 128, 192, 3)
             
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
                flow_3ch = np.dstack((flow, mag)) # (H, W, 3)
                
                all_images.append(processed_frames_arr[t]) 
                all_flows.append(flow_3ch)
                
    return np.array(all_images), np.array(all_flows)

import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

def load_mm_data(mm_training_dir, csv_path, target_size=(128, 192)):
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
                frame_resized = cv2.resize(
                    frame.astype(np.float32),
                    (target_size[1], target_size[0])
                )
                
                # Normalize using the percentiles
                if p99 - p1 == 0:
                    frame_norm = np.zeros(target_size)
                else:
                    frame_norm = (frame_resized - p1) / slice_range
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
                flow_3ch = np.dstack((flow, mag))  # (H, W, 3)

                all_images.append(processed_frames_arr[t])
                all_flows.append(flow_3ch)

    return np.array(all_images), np.array(all_flows)


def load_combined_data(acdc_dir, mm_training_dir, csv_path, target_size=(128, 192)):
    """
    Load and concatenate NOR patients from both ACDC (Dataset_2) and
    M&M (Dataset_1) datasets.

    Parameters
    ----------
    acdc_dir       : str    root of Dataset_2 (contains 'database/training')
    mm_training_dir: str    Dataset_1/Training folder
    csv_path       : str    M&M CSV metadata file path
    target_size    : tuple  (H, W) resize target, default (128, 192)

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
