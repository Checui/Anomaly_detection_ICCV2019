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
# IMPORT YOUR NEW DATA LOADER FILE HERE
from data_loader import load_acdc_data, load_mm_data, load_combined_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ACDC', 'MM', 'COMBINED'])
    parser.add_argument('--acdc_dir', type=str, default='./Dataset_2')
    parser.add_argument('--mm_dir', type=str, default='./Dataset_1/Training')
    parser.add_argument('--mm_csv', type=str, default='./Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # 1. Load the data based on the argument
    if args.dataset == 'ACDC':
        images, flows = load_acdc_data(args.acdc_dir)
        dataset_name = 'ACDC_NOR'
        
    elif args.dataset == 'MM':
        images, flows = load_mm_data(args.mm_dir, args.mm_csv)
        dataset_name = 'MM_NOR'
        
    elif args.dataset == 'COMBINED':
        images, flows = load_combined_data(args.acdc_dir, args.mm_dir, args.mm_csv)
        dataset_name = 'COMBINED_NOR'

    print(f"Loaded {len(images)} samples for {args.dataset}.")
    
    # 2. Train the model
    if len(images) > 0:
        tf.compat.v1.reset_default_graph() 
        
        print(f"Starting Training for {args.dataset}...")
        GAN_tf.train_Unet_naive_with_batch_norm(
            training_images=images,
            training_flows=flows,
            max_epoch=args.epochs,
            dataset_name=dataset_name,
            batch_size=4
        )
        print(f"Training complete for {args.dataset}.")
    else:
        print("No data loaded. Check paths.")