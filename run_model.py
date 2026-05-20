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

import GAN_tf
import GAN_tf_rgb
import numpy as np
import data_loader     as dl_flow   # returns (images, flows)
import data_loader_rgb as dl_rgb    # returns (es_images, ed_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Model type ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--model_type', type=str, default='flow',
        choices=['flow', 'rgb'],
        help=(
            '"flow" (default): second decoder head predicts optical flow '
            '(GAN_tf); first decoder always reconstructs the input frame. '
            '"rgb": second decoder head predicts the ED frame from the ES '
            'frame instead of optical flow (GAN_tf_rgb).'
        )
    )

    # ── Dataset / frame mode ──────────────────────────────────────────────────
    parser.add_argument(
        '--datasets', type=str, nargs='+', required=True,
        choices=['ACDC', 'MM', 'RECON'],
        help='Training datasets to load (space-separated). Choices: ACDC MM RECON.'
    )
    parser.add_argument(
        '--val_datasets', type=str, nargs='+', default=['ACDC', 'MM'],
        choices=['ACDC', 'MM'],
        help=(
            'Validation datasets to load (space-separated). Choices: ACDC MM. '
            'Defaults to "ACDC MM" (combined). RECON is excluded — every RECON '
            'subject is NOR, so it cannot supply positive samples for AUC.'
        )
    )
    parser.add_argument(
        '--frame_mode', type=str, default='ed_es',
        choices=['ed_es', 'next_frame', 'next_frame_systole'],
        help=(
            '"ed_es" (default): use only End-Diastole/End-Systole frame pairs. '
            '"next_frame": use all consecutive frame pairs. '
            '"next_frame_systole": use consecutive pairs (t, t+1) for '
            't in [ED, ES-1] only (systolic contraction phase). Patients '
            'with ES <= ED are skipped.'
        )
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument('--acdc_dir',    type=str, default='../Dataset_2')
    parser.add_argument('--mm_dir',      type=str, default='../Dataset_1/Training')
    parser.add_argument('--mm_val_dir',  type=str, default='../Dataset_1/Validation')
    parser.add_argument('--mm_csv',      type=str,
                        default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    parser.add_argument('--recon_dir',   type=str, default=None,
                        help='Path to reconstructed_sax_images_training_2023/. '
                             'Required when RECON is in --datasets.')
    parser.add_argument('--recon_csv',   type=str, default=None,
                        help='Path to ed_es_frames.csv (only used for RECON + ed_es). '
                             'Defaults to <recon_dir>/segmentation/ed_es_frames.csv.')

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to resume training from.')
    parser.add_argument('--run_tag',     type=str, default=None,
                        help='Optional suffix appended to the checkpoint folder name '
                             'to distinguish re-runs (e.g. --run_tag v2).')
    parser.add_argument('--lw_adv',  type=float, default=0.25,
                        help='Loss weight for the adversarial term in G_loss_total.')
    parser.add_argument('--lw_appe', type=float, default=1.0,
                        help='Loss weight for the appearance reconstruction term.')
    parser.add_argument('--lw_aux',  type=float, default=2.0,
                        help='Loss weight for the auxiliary prediction term (flow or ED).')

    args = parser.parse_args()

    if 'RECON' in args.datasets and args.recon_dir is None:
        parser.error('--recon_dir is required when RECON is in --datasets')

    # ------------------------------------------------------------------
    # 1. Build per-dataset loader dispatch tables
    # ------------------------------------------------------------------
    # Both tables return (part1, part2) where:
    #   flow mode  → part1 = images (ES frames),  part2 = flows
    #   rgb  mode  → part1 = es_images,            part2 = ed_images

    def _resolve_recon_csv():
        return args.recon_csv or os.path.join(
            args.recon_dir, 'segmentation', 'ed_es_frames.csv')

    def _recon_flow_ed_es():
        return dl_flow.load_reconstructed_sax_data(args.recon_dir, _resolve_recon_csv())

    def _recon_rgb_ed_es():
        return dl_rgb.load_reconstructed_sax_data_rgb(args.recon_dir, _resolve_recon_csv())

    def _recon_flow_systole():
        return dl_flow.load_reconstructed_sax_data_next_frame(
            args.recon_dir, restrict_to_systole=True, ed_es_csv=_resolve_recon_csv())

    def _recon_rgb_systole():
        return dl_rgb.load_reconstructed_sax_data_next_frame_rgb(
            args.recon_dir, restrict_to_systole=True, ed_es_csv=_resolve_recon_csv())

    _flow_loaders = {
        ('ACDC',  'ed_es'):              lambda: dl_flow.load_acdc_ed_es_data(args.acdc_dir),
        ('ACDC',  'next_frame'):         lambda: dl_flow.load_acdc_data(args.acdc_dir),
        ('ACDC',  'next_frame_systole'): lambda: dl_flow.load_acdc_data(args.acdc_dir, restrict_to_systole=True),
        ('MM',    'ed_es'):              lambda: dl_flow.load_mm_ed_es_data(args.mm_dir, args.mm_csv),
        ('MM',    'next_frame'):         lambda: dl_flow.load_mm_data(args.mm_dir, args.mm_csv),
        ('MM',    'next_frame_systole'): lambda: dl_flow.load_mm_data(args.mm_dir, args.mm_csv, restrict_to_systole=True),
        ('RECON', 'ed_es'):              _recon_flow_ed_es,
        ('RECON', 'next_frame'):         lambda: dl_flow.load_reconstructed_sax_data_next_frame(args.recon_dir),
        ('RECON', 'next_frame_systole'): _recon_flow_systole,
    }

    _rgb_loaders = {
        ('ACDC',  'ed_es'):              lambda: dl_rgb.load_acdc_ed_es_data(args.acdc_dir),
        ('ACDC',  'next_frame'):         lambda: dl_rgb.load_acdc_data(args.acdc_dir),
        ('ACDC',  'next_frame_systole'): lambda: dl_rgb.load_acdc_data(args.acdc_dir, restrict_to_systole=True),
        ('MM',    'ed_es'):              lambda: dl_rgb.load_mm_ed_es_data(args.mm_dir, args.mm_csv),
        ('MM',    'next_frame'):         lambda: dl_rgb.load_mm_data(args.mm_dir, args.mm_csv),
        ('MM',    'next_frame_systole'): lambda: dl_rgb.load_mm_data(args.mm_dir, args.mm_csv, restrict_to_systole=True),
        ('RECON', 'ed_es'):              _recon_rgb_ed_es,
        ('RECON', 'next_frame'):         lambda: dl_rgb.load_reconstructed_sax_data_next_frame_rgb(args.recon_dir),
        ('RECON', 'next_frame_systole'): _recon_rgb_systole,
    }

    loaders = _flow_loaders if args.model_type == 'flow' else _rgb_loaders

    # ------------------------------------------------------------------
    # 2. Load TRAINING data
    # ------------------------------------------------------------------
    part1_list, part2_list = [], []

    for ds in args.datasets:
        print(f"\n=== Loading {ds} [{args.frame_mode}] ({args.model_type} mode) ===")
        p1, p2 = loaders[(ds, args.frame_mode)]()
        print(f"{ds}: {len(p1)} samples loaded.")
        if len(p1) > 0:
            part1_list.append(p1)
            part2_list.append(p2)

    if not part1_list:
        print("No data loaded. Check paths and dataset flags.")
        sys.exit(1)

    train_part1 = np.concatenate(part1_list, axis=0)
    train_part2 = np.concatenate(part2_list, axis=0)

    dataset_name = '_'.join(args.datasets) + '_' + args.frame_mode.upper() + '_' + args.model_type.upper() + '_NOR'
    if args.run_tag:
        dataset_name = dataset_name + '_' + args.run_tag
    print(f"\nTotal training samples: {len(train_part1)}")
    print(f"Dataset name: {dataset_name}")

    # ------------------------------------------------------------------
    # 3. Load VALIDATION data (matches frame_mode: ed_es or next_frame)
    # ------------------------------------------------------------------
    val_mode_label = {
        'ed_es': 'ED/ES',
        'next_frame': 'next_frame',
        'next_frame_systole': 'next_frame (systole only)',
    }[args.frame_mode]
    print(f"\n=== Loading Validation Data ({val_mode_label}) ===")
    print(f"Validation datasets: {' '.join(args.val_datasets)}")

    if args.frame_mode == 'next_frame':
        if args.model_type == 'flow':
            _val_acdc = dl_flow.load_acdc_test_val_data
            _val_mm   = dl_flow.load_mm_validation_data
        else:
            _val_acdc = dl_rgb.load_acdc_test_val_data
            _val_mm   = dl_rgb.load_mm_validation_data
    elif args.frame_mode == 'next_frame_systole':
        if args.model_type == 'flow':
            _val_acdc = lambda d: dl_flow.load_acdc_test_val_data(d, restrict_to_systole=True)
            _val_mm   = lambda d, c: dl_flow.load_mm_validation_data(d, c, restrict_to_systole=True)
        else:
            _val_acdc = lambda d: dl_rgb.load_acdc_test_val_data(d, restrict_to_systole=True)
            _val_mm   = lambda d, c: dl_rgb.load_mm_validation_data(d, c, restrict_to_systole=True)
    else:
        if args.model_type == 'flow':
            _val_acdc = dl_flow.load_acdc_test_val_ed_es_data
            _val_mm   = dl_flow.load_mm_validation_ed_es_data
        else:
            _val_acdc = dl_rgb.load_acdc_test_val_ed_es_data
            _val_mm   = dl_rgb.load_mm_validation_ed_es_data

    val_parts_p1, val_parts_p2, val_labels, val_pids, val_slice_idxs = [], [], [], [], []

    if 'ACDC' in args.val_datasets:
        # Use the ENTIRE ACDC test set (all 50 patients) for validation —
        # the original 12-patient val split was not representative enough.
        # The loader still returns the same val/test split for reproducibility;
        # here we simply concatenate both halves into one validation set.
        (acdc_val_p1, acdc_val_p2, acdc_val_labels, acdc_val_pids, acdc_val_slcs,
         acdc_test_p1, acdc_test_p2, acdc_test_labels, acdc_test_pids, acdc_test_slcs
         ) = _val_acdc(args.acdc_dir)
        if len(acdc_test_p1) > 0:
            acdc_val_p1 = np.concatenate([acdc_val_p1, acdc_test_p1], axis=0)
            acdc_val_p2 = np.concatenate([acdc_val_p2, acdc_test_p2], axis=0)
            acdc_val_labels = list(acdc_val_labels) + list(acdc_test_labels)
            acdc_val_pids   = list(acdc_val_pids)   + list(acdc_test_pids)
            acdc_val_slcs   = list(acdc_val_slcs)   + list(acdc_test_slcs)
        # Disambiguate ACDC vs M&M pids that might collide
        acdc_val_pids = [f"ACDC_{pid}" for pid in acdc_val_pids]
        print(f"ACDC validation ({val_mode_label}): {len(acdc_val_p1)} samples "
              f"from {len(set(acdc_val_pids))} patients (full test set)")
        if len(acdc_val_p1) > 0:
            val_parts_p1.append(acdc_val_p1)
            val_parts_p2.append(acdc_val_p2)
            val_labels.extend(acdc_val_labels)
            val_pids.extend(acdc_val_pids)
            val_slice_idxs.extend(acdc_val_slcs)

    if 'MM' in args.val_datasets:
        mm_val_p1, mm_val_p2, mm_val_labels, mm_val_pids, mm_val_slcs = _val_mm(
            args.mm_val_dir, args.mm_csv)
        mm_val_pids = [f"MM_{pid}" for pid in mm_val_pids]
        print(f"M&M validation ({val_mode_label}):  {len(mm_val_p1)} samples")
        if len(mm_val_p1) > 0:
            val_parts_p1.append(mm_val_p1)
            val_parts_p2.append(mm_val_p2)
            val_labels.extend(mm_val_labels)
            val_pids.extend(mm_val_pids)
            val_slice_idxs.extend(mm_val_slcs)

    if val_parts_p1:
        val_p1 = np.concatenate(val_parts_p1, axis=0)
        val_p2 = np.concatenate(val_parts_p2, axis=0)
    else:
        val_p1, val_p2, val_labels, val_pids, val_slice_idxs = None, None, None, None, None

    if val_p1 is not None:
        from collections import Counter
        lbl_counts = Counter(val_labels)
        print(f"Combined validation: {len(val_p1)} samples")
        print(f"  Healthy (NOR): {lbl_counts.get('NOR', 0)}, "
              f"Unhealthy: {sum(v for k, v in lbl_counts.items() if k != 'NOR')}")
    else:
        print("WARNING: No validation data loaded!")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    tf.compat.v1.reset_default_graph()

    print(f"\nStarting Training: {dataset_name} ...")

    if args.model_type == 'flow':
        GAN_tf.train_Unet_naive_with_batch_norm(
            training_images=train_part1,
            training_flows=train_part2,
            max_epoch=args.epochs,
            dataset_name=dataset_name,
            start_model_idx=args.start_epoch,
            batch_size=16,
            val_images=val_p1,
            val_flows=val_p2,
            val_labels=val_labels,
            val_pids=val_pids,
            val_slice_idxs=val_slice_idxs,
            lw_adv=args.lw_adv,
            lw_appe=args.lw_appe,
            lw_aux=args.lw_aux,
        )
    else:
        GAN_tf_rgb.train_Unet_naive_with_batch_norm(
            training_es_images=train_part1,
            training_ed_images=train_part2,
            max_epoch=args.epochs,
            dataset_name=dataset_name,
            start_model_idx=args.start_epoch,
            batch_size=16,
            val_es_images=val_p1,
            val_ed_images=val_p2,
            val_labels=val_labels,
            val_pids=val_pids,
            val_slice_idxs=val_slice_idxs,
            lw_adv=args.lw_adv,
            lw_appe=args.lw_appe,
            lw_aux=args.lw_aux,
        )

    print(f"Training complete: {dataset_name}.")