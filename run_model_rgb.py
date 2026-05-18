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
    load_acdc_ed_es_data,
    load_mm_data,
    load_mm_ed_es_data,
    load_reconstructed_sax_data_rgb,
    load_reconstructed_sax_data_next_frame_rgb,
    load_acdc_test_val_ed_es_data,
    load_mm_validation_ed_es_data,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', type=str, nargs='+', required=True,
        choices=['ACDC', 'MM', 'RECON'],
        help='Training datasets to load (space-separated). Choices: ACDC MM RECON. '
             'Example: --datasets ACDC MM RECON'
    )
    parser.add_argument(
        '--frame_mode', type=str, default='ed_es',
        choices=['ed_es', 'next_frame'],
        help='Frame pair strategy for training: '
             '"ed_es" uses only End-Diastole/End-Systole pairs (default); '
             '"next_frame" uses all consecutive frame pairs.'
    )
    parser.add_argument('--acdc_dir', type=str, default='../Dataset_2')
    parser.add_argument('--mm_dir', type=str, default='../Dataset_1/Training')
    parser.add_argument('--mm_val_dir', type=str, default='../Dataset_1/Validation')
    parser.add_argument('--mm_csv', type=str,
                        default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    parser.add_argument('--recon_dir', type=str, default=None,
                        help='Path to reconstructed_sax_images_training_2023/ folder. '
                             'Required when RECON is included in --datasets.')
    parser.add_argument('--recon_csv', type=str, default=None,
                        help='Path to ed_es_frames.csv; defaults to '
                             '<recon_dir>/segmentation/ed_es_frames.csv. '
                             'Only used in ed_es mode for RECON.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to resume training from')

    args = parser.parse_args()

    # Validate: RECON requires --recon_dir
    if 'RECON' in args.datasets and args.recon_dir is None:
        parser.error('--recon_dir is required when RECON is included in --datasets')

    # ------------------------------------------------------------------
    # 1. Load TRAINING data
    # ------------------------------------------------------------------
    def _load_recon_ed_es():
        recon_csv = args.recon_csv or os.path.join(
            args.recon_dir, 'segmentation', 'ed_es_frames.csv')
        return load_reconstructed_sax_data_rgb(args.recon_dir, recon_csv)

    loaders = {
        ('ACDC',  'ed_es'):      lambda: load_acdc_ed_es_data(args.acdc_dir),
        ('ACDC',  'next_frame'): lambda: load_acdc_data(args.acdc_dir),
        ('MM',    'ed_es'):      lambda: load_mm_ed_es_data(args.mm_dir, args.mm_csv),
        ('MM',    'next_frame'): lambda: load_mm_data(args.mm_dir, args.mm_csv),
        ('RECON', 'ed_es'):      _load_recon_ed_es,
        ('RECON', 'next_frame'): lambda: load_reconstructed_sax_data_next_frame_rgb(args.recon_dir),
    }

    all_es_parts = []
    all_ed_parts = []

    for ds in args.datasets:
        print(f"\n=== Loading {ds} [{args.frame_mode}] ===")
        es_imgs, ed_imgs = loaders[(ds, args.frame_mode)]()
        print(f"{ds}: {len(es_imgs)} samples loaded.")
        if len(es_imgs) > 0:
            all_es_parts.append(es_imgs)
            all_ed_parts.append(ed_imgs)

    if not all_es_parts:
        print("No data loaded. Check paths and dataset flags.")
        sys.exit(1)

    es_images = np.concatenate(all_es_parts, axis=0)
    ed_images = np.concatenate(all_ed_parts, axis=0)

    dataset_name = '_'.join(args.datasets) + '_' + args.frame_mode.upper() + '_NOR'
    print(f"\nTotal training samples: {len(es_images)}")
    print(f"Dataset name: {dataset_name}")

    # ------------------------------------------------------------------
    # 2. Load VALIDATION data (always ED/ES regardless of frame_mode)
    # ------------------------------------------------------------------
    print("\n=== Loading Validation Data (ED/ES) ===")

    (acdc_val_es, acdc_val_ed, acdc_val_labels, acdc_val_pids, _acdc_val_slcs,
     _, _, _, _, _) = load_acdc_test_val_ed_es_data(args.acdc_dir)
    print(f"ACDC validation (ED/ES): {len(acdc_val_es)} samples")

    mm_val_es, mm_val_ed, mm_val_labels, mm_val_pids, _mm_val_slcs = load_mm_validation_ed_es_data(
        args.mm_val_dir, args.mm_csv
    )
    print(f"M&M validation (ED/ES):  {len(mm_val_es)} samples")

    if len(acdc_val_es) > 0 and len(mm_val_es) > 0:
        val_es_images = np.concatenate([acdc_val_es, mm_val_es], axis=0)
        val_ed_images = np.concatenate([acdc_val_ed, mm_val_ed], axis=0)
        val_labels    = acdc_val_labels + mm_val_labels
    elif len(acdc_val_es) > 0:
        val_es_images, val_ed_images, val_labels = acdc_val_es, acdc_val_ed, acdc_val_labels
    elif len(mm_val_es) > 0:
        val_es_images, val_ed_images, val_labels = mm_val_es, mm_val_ed, mm_val_labels
    else:
        val_es_images, val_ed_images, val_labels = None, None, None

    if val_es_images is not None:
        from collections import Counter
        lbl_counts = Counter(val_labels)
        print(f"Combined validation: {len(val_es_images)} samples")
        print(f"  Healthy (NOR): {lbl_counts.get('NOR', 0)}, "
              f"Unhealthy: {sum(v for k, v in lbl_counts.items() if k != 'NOR')}")
    else:
        print("WARNING: No validation data loaded!")

    # ------------------------------------------------------------------
    # 3. Train the model
    # ------------------------------------------------------------------
    tf.compat.v1.reset_default_graph()

    print(f"\nStarting Training: {dataset_name} ...")
    GAN_tf_rgb.train_Unet_naive_with_batch_norm(
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
    print(f"Training complete: {dataset_name}.")