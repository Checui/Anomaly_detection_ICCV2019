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
import GAN_tf
import numpy as np
from data_loader import (
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
        images, flows = load_acdc_data(args.acdc_dir)
        dataset_name = 'ACDC_NOR'
        
    elif args.dataset == 'MM':
        images, flows = load_mm_data(args.mm_dir, args.mm_csv)
        dataset_name = 'MM_NOR'
        
    elif args.dataset == 'COMBINED':
        images, flows = load_combined_ed_es_data(args.acdc_dir, args.mm_dir, args.mm_csv)
        dataset_name = 'COMBINED_NOR'

    print(f"Loaded {len(images)} training samples for {args.dataset}.")

    # 2. Load the VALIDATION data (ACDC test-split + M&M Validation, all pathologies)
    print("\n=== Loading Validation Data ===")

    # ACDC validation split (4 NOR + 2 per disease = 12 patients, ED/ES only)
    (acdc_val_images, acdc_val_flows, acdc_val_labels, acdc_val_pids,
     _, _, _, _) = load_acdc_test_val_ed_es_data(args.acdc_dir)
    print(f"ACDC validation (ED/ES): {len(acdc_val_images)} samples")

    # M&M validation (all pathologies, ED/ES only)
    mm_val_images, mm_val_flows, mm_val_labels, mm_val_pids = load_mm_validation_ed_es_data(
        args.mm_val_dir, args.mm_csv
    )
    print(f"M&M validation (ED/ES):  {len(mm_val_images)} samples")

    # Combine validation sets
    if len(acdc_val_images) > 0 and len(mm_val_images) > 0:
        val_images = np.concatenate([acdc_val_images, mm_val_images], axis=0)
        val_flows  = np.concatenate([acdc_val_flows,  mm_val_flows],  axis=0)
    elif len(acdc_val_images) > 0:
        val_images, val_flows = acdc_val_images, acdc_val_flows
    elif len(mm_val_images) > 0:
        val_images, val_flows = mm_val_images, mm_val_flows
    else:
        val_images, val_flows = None, None

    if val_images is not None:
        print(f"Combined validation: {len(val_images)} samples")
    else:
        print("WARNING: No validation data loaded!")

    # 3. Train the model
    if len(images) > 0:
        tf.compat.v1.reset_default_graph() 
        
        print(f"\nStarting Training for {args.dataset}...")
        GAN_tf.train_Unet_naive_with_batch_norm(
            training_images=images,
            training_flows=flows,
            max_epoch=args.epochs,
            dataset_name=dataset_name,
            start_model_idx=args.start_epoch,
            batch_size=4,
            val_images=val_images,
            val_flows=val_flows
        )
        print(f"Training complete for {args.dataset}.")
    else:
        print("No data loaded. Check paths.")
