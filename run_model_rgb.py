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

# Import your local modules
import GAN_tf_rgb
import numpy as np
from data_loader_rgb import (
    load_acdc_data,
    load_mm_data,
    load_combined_data,
    load_combined_ed_es_data,
    load_acdc_test_val_ed_es_data,
    load_mm_validation_ed_es_data,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ACDC', 'MM', 'COMBINED'])
    parser.add_argument('--acdc_dir', type=str, default='../Dataset_2')
    parser.add_argument('--mm_dir', type=str, default='../Dataset_1/Training')
    parser.add_argument('--mm_val_dir', type=str, default='../Dataset_1/Validation')
    parser.add_argument('--mm_csv', type=str, default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to resume training from')

    args = parser.parse_args()

    # 1. Load the TRAINING data based on the argument
    if args.dataset == 'ACDC':
        es_images, ed_images = load_acdc_data(args.acdc_dir)
        dataset_name = 'ACDC_NOR'
        
    elif args.dataset == 'MM':
        es_images, ed_images = load_mm_data(args.mm_dir, args.mm_csv)
        dataset_name = 'MM_NOR'
        
    elif args.dataset == 'COMBINED':
        es_images, ed_images = load_combined_ed_es_data(args.acdc_dir, args.mm_dir, args.mm_csv)
        dataset_name = 'COMBINED_NOR'

    print(f"Loaded {len(es_images)} training samples for {args.dataset}.")

    # 2. Load the VALIDATION data (ACDC test-split + M&M Validation, all pathologies)
    print("\n=== Loading Validation Data ===")

    # ACDC validation split (4 NOR + 2 per disease = 12 patients, ED/ES only)
    (acdc_val_es_images, acdc_val_ed_images, acdc_val_labels, acdc_val_pids,
     _, _, _, _) = load_acdc_test_val_ed_es_data(args.acdc_dir)
    print(f"ACDC validation (ED/ES): {len(acdc_val_es_images)} samples")

    # M&M validation (all pathologies, ED/ES only)
    mm_val_es_images, mm_val_ed_images, mm_val_labels, mm_val_pids = load_mm_validation_ed_es_data(
        args.mm_val_dir, args.mm_csv
    )
    print(f"M&M validation (ED/ES):  {len(mm_val_es_images)} samples")

    # Combine validation sets
    if len(acdc_val_es_images) > 0 and len(mm_val_es_images) > 0:
        val_es_images = np.concatenate([acdc_val_es_images, mm_val_es_images], axis=0)
        val_ed_images = np.concatenate([acdc_val_ed_images, mm_val_ed_images], axis=0)
        val_labels = acdc_val_labels + mm_val_labels
    elif len(acdc_val_es_images) > 0:
        val_es_images, val_ed_images, val_labels = acdc_val_es_images, acdc_val_ed_images, acdc_val_labels
    elif len(mm_val_es_images) > 0:
        val_es_images, val_ed_images, val_labels = mm_val_es_images, mm_val_ed_images, mm_val_labels
    else:
        val_es_images, val_ed_images, val_labels = None, None, None

    if val_es_images is not None:
        from collections import Counter
        lbl_counts = Counter(val_labels)
        print(f"Combined validation: {len(val_es_images)} samples")
        print(f"  Healthy (NOR): {lbl_counts.get('NOR', 0)}, Unhealthy: {sum(v for k,v in lbl_counts.items() if k != 'NOR')}")
    else:
        print("WARNING: No validation data loaded!")

    # 3. Train the model
    if len(es_images) > 0:
        tf.compat.v1.reset_default_graph() 
        
        print(f"\nStarting Training for {args.dataset}...")
        
        # KEY FIX: Match the keyword arguments to the new GAN_tf.py signature
        GAN_tf.train_Unet_naive_with_batch_norm(
            training_es_images=es_images,
            training_ed_images=ed_images,
            max_epoch=args.epochs,
            dataset_name=dataset_name,
            start_model_idx=args.start_epoch,
            batch_size=16,
            val_es_images=val_es_images,
            val_ed_images=val_ed_images,
            val_labels=val_labels
        )
        print(f"Training complete for {args.dataset}.")
    else:
        print("No data loaded. Check paths.")