import sys
import os
import argparse

# Mock ProgressBar (if needed for your GAN_tf code)
import types
class MockProgressBar:
    FULL = 'full'
    def __init__(self, n, fmt='full'):
        self.n = n
        self.current = 0
    def __call__(self): pass
    def done(self): pass

mock_pb_module = types.ModuleType("ProgressBar")
mock_pb_module.ProgressBar = MockProgressBar
sys.modules["ProgressBar"] = mock_pb_module

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf

import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
import traceback

# IMPORTANT: Ensure your GAN_tf module is in the same directory on the HPC
import GAN_tf 

# ── ED/ES-only combined data loader ──────────────────────────────────────────
def load_combined_ed_es_data(acdc_dir, mm_training_dir, csv_path, target_size=(128, 192)):
    """
    Load only End-Diastole (ED) and End-Systole (ES) frame pairs
    from both ACDC (Dataset_2) and M&M (Dataset_1) NOR subjects.
    """
    def _preprocess_frame(frame, p1, p99, target_size):
        """Resize + percentile-normalise a single 2-D frame to RGB."""
        frame_resized = cv2.resize(
            frame.astype(np.float32),
            (target_size[1], target_size[0])
        )
        if (p99 - p1) < 1e-7:
            frame_norm = np.zeros(target_size, dtype=np.float32)
        else:
            frame_norm = np.clip((frame_resized - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        return np.stack([frame_norm] * 3, axis=-1)  

    all_images = []
    all_flows  = []

    # ── ACDC ────────────────────────────────────────────────────────────────
    training_dir = os.path.join(acdc_dir, 'database', 'training')
    if not os.path.isdir(training_dir):
        training_dir = os.path.join(acdc_dir, 'database', 'training_test')
    
    patients = sorted([d for d in os.listdir(training_dir)
                       if os.path.isdir(os.path.join(training_dir, d))])
    print(f"ACDC: found {len(patients)} patient folders.")

    for p in patients:
        p_dir    = os.path.join(training_dir, p)
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
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path)) 
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
            slice_seq = img_arr[:, z, :, :] 
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
            flow_3ch = np.dstack((flow, mag)) 

            all_images.append(es_frame_rgb)
            all_flows.append(flow_3ch)

    print(f"ACDC NOR ED/ES samples: {len(all_images)}")
    acdc_count = len(all_images)

    # ── M&M ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
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
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path)) 
        except Exception as e:
            print(f"Load error {subject_id}: {e}")
            continue

        if img_arr.ndim != 4:
            continue
        T, Z, _, _ = img_arr.shape

        if ed_idx == es_idx:
            print(f"Skipping M&M {subject_id}: ED and ES frames are identical ({ed_idx})")
            continue

        if es_idx >= T or ed_idx >= T:
            print(f"Skipping {subject_id}: ED={ed_idx} or ES={es_idx} out of range T={T}")
            continue

        print(f"M&M {subject_id} (NOR)  ED={ed_idx}  ES={es_idx}")
        for z in range(Z):
            slice_seq = img_arr[:, z, :, :] 
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
            flow_3ch = np.dstack((flow, mag))

            all_images.append(es_frame_rgb)
            all_flows.append(flow_3ch)

    mm_count = len(all_images) - acdc_count
    print(f"M&M NOR ED/ES samples: {mm_count}")
    print(f"Combined ED/ES samples: {len(all_images)}")

    if len(all_images) == 0:
        return np.array([]), np.array([])
    return np.array(all_images), np.array(all_flows)


# ── Main Execution Block ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Setup Argparse to catch inputs from submit_job.pbs
    parser = argparse.ArgumentParser(description="Train Combined Models on ACDC and M&Ms")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset identifier (e.g., COMBINED)")
    parser.add_argument('--acdc_dir', type=str, required=True, help="Path to ACDC dataset root")
    parser.add_argument('--mm_dir', type=str, required=True, help="Path to M&M training dataset")
    parser.add_argument('--mm_csv', type=str, required=True, help="Path to M&M metadata CSV")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()

    print(f"--- Starting Model Training ---")

    try:
        # 2. Load the data using arguments
        images_ed_es, flows_ed_es = load_combined_ed_es_data(
            acdc_dir=args.acdc_dir, 
            mm_training_dir=args.mm_dir, 
            csv_path=args.mm_csv
        )
        print(f"\nLoaded {len(images_ed_es)} ED/ES samples in total.")

        if len(images_ed_es) > 0:
            print(f"Images shape : {images_ed_es.shape}")
            print(f"Flows  shape : {flows_ed_es.shape}")

            tf.compat.v1.reset_default_graph()

            print("\nStarting Training (ED/ES combined, ES→ED)...")
            
            # 3. Pass the epochs argument from the PBS file to your trainer
            GAN_tf.train_Unet_naive_with_batch_norm(
                training_images=images_ed_es,
                training_flows=flows_ed_es,
                max_epoch=args.epochs,  
                dataset_name=f'ACDC_MM_EDES_{args.dataset}', 
                batch_size=4
            )
            print("Training complete.")
        else:
            print("No data loaded. Check dataset paths and NOR filtering.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()