"""
Final Evaluation Script — Test Split Only for BraTS 2023 Adult Glioma Dataset.

Loads the 20 % test-split subjects from the shared SplitManager manifest
(the same subjects that were locked away during pre-training AND fine-tuning)
and evaluates the best fine-tuned model against them.

Data-safety guarantee:
  • Reads the manifest → test_ids (only those subjects were never seen during
    training or validation).
  • SplitManager.assert_no_leakage() is called before inference.
  • Prints a per-subject results table plus a final aggregated summary.
  • Optionally exports predicted segmentation masks as NIfTI files.
"""

from __future__ import annotations
import sys
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.models import SwinMambaDual
from src.data import BraTS2023GliomaDataset, SplitManager
from src.utils import (
    evaluate_segmentation_case,
    compute_classification_metrics,
    postprocess_predictions,
    CheckpointManager,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SwinMamba on BraTS 2023 Test Split")
    p.add_argument("--data_dir",      type=str, default=r"C:\Glioma\Dataset\Glioma")
    p.add_argument("--manifest_dir",  type=str, default=r"C:\Glioma\checkpoints\split",
                   help="Must match the manifest_dir used in pretrain.py / finetune.py")
    p.add_argument("--no_checksums",  action="store_true")

    p.add_argument("--checkpoint_dir", type=str, default=r"C:\Glioma\checkpoints\finetune",
                   help="Fine-tune checkpoint directory (loads finetune_best.pt)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Override: path to a specific .pt file to evaluate")

    p.add_argument("--img_size",      type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--save_predictions", action="store_true",
                   help="Export predicted NIfTI segmentation masks to --output_dir")
    p.add_argument("--output_dir",    type=str, default=r"C:\Glioma\outputs\predictions")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== [Evaluation] Dataset: BraTS 2023 Adult Glioma | device={device} ===\n")

    # ── 1. Load split manifest and extract TEST subjects ──────────────────
    sm = SplitManager(
        data_dir          = args.data_dir,
        manifest_dir      = args.manifest_dir,
        force_rebuild     = False,
        compute_checksums = not args.no_checksums,
    )
    sm.assert_no_leakage()

    train_ids, val_ids, test_ids = sm.get_splits()
    test_subjects = [sm.get_subject(sid) for sid in test_ids]

    print(
        f"Split recap: {len(train_ids)} train | {len(val_ids)} val | "
        f"{len(test_ids)} test\n"
        f"Evaluating on {len(test_subjects)} TEST subjects ONLY.\n"
    )

    # ── 2. Dataset & DataLoader ───────────────────────────────────────────
    test_ds = BraTS2023GliomaDataset(
        subjects     = test_subjects,
        mode         = "finetune",
        target_shape = tuple(args.img_size),
        augment      = False,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = SwinMambaDual(
        in_chans      = 4,
        seg_out_chans = 3,
        num_classes   = 2,
        feat_size     = (48, 96, 192, 384),
        hidden_size   = 768,
        mode          = "finetune",
    ).to(device)

    # ── 4. Load checkpoint ────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        checkpoint_dir = args.checkpoint_dir,
        stage          = "finetune",
        split_manager  = sm,
        metric_name    = "mean_dice",
        metric_mode    = "max",
    )

    if args.checkpoint_path:
        ckpt_mgr.load_from_path(args.checkpoint_path, model)
    else:
        ckpt_mgr.load_best(model)

    model.eval()
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── 5. Inference ──────────────────────────────────────────────────────
    all_metrics = []
    clf_preds   = []
    clf_gts     = []

    header = (
        f"{'Subject':<32} | {'Dice-WT':>7} {'Dice-TC':>7} {'Dice-ET':>7} "
        f"{'MeanDice':>8} {'HD95':>6} | {'Pred':>5} {'GT':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Cases"):
            sub_id   = batch["subject_id"][0]
            imgs     = batch["image"].to(device)
            seg_gt   = batch["seg"].numpy()[0]
            label_gt = batch["label"].item()

            outs     = model(imgs)
            seg_raw  = outs["seg"][0].float().cpu().numpy()
            clf_pred = torch.argmax(outs["clf"][0]).item()

            clf_preds.append(clf_pred)
            clf_gts.append(label_gt)

            seg_bin  = postprocess_predictions(seg_raw)
            metrics  = evaluate_segmentation_case(seg_bin, seg_gt)
            all_metrics.append(metrics)

            grade_str  = lambda v: "HGG" if v == 1 else "LGG"
            print(
                f"{sub_id:<32} | "
                f"{metrics['dice_wt']:>7.4f} {metrics['dice_tc']:>7.4f} "
                f"{metrics['dice_et']:>7.4f} {metrics['mean_dice']:>8.4f} "
                f"{metrics['mean_hd95']:>6.1f} | "
                f"{grade_str(clf_pred):>5} {grade_str(label_gt):>5}"
            )

            if args.save_predictions:
                out_path = os.path.join(args.output_dir, f"{sub_id}_seg_pred.nii.gz")
                arr = seg_bin.transpose(1, 2, 3, 0).astype(np.float32)
                nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), out_path)

    # ── 6. Aggregate Summary ──────────────────────────────────────────────
    clf_res = compute_classification_metrics(clf_preds, clf_gts)
    n       = len(all_metrics)

    print(f"\n{sep}")
    print("  FINAL TEST RESULTS (BraTS 2023 Adult Glioma)")
    print(sep)
    print(f"  Subjects evaluated   : {n}")
    print(f"  Dice - Mean (all)    : {np.mean([m['mean_dice'] for m in all_metrics]):.4f}")
    print(f"  Dice - Whole Tumor   : {np.mean([m['dice_wt']   for m in all_metrics]):.4f}")
    print(f"  Dice - Tumor Core    : {np.mean([m['dice_tc']   for m in all_metrics]):.4f}")
    print(f"  Dice - Enhancing     : {np.mean([m['dice_et']   for m in all_metrics]):.4f}")
    print(f"  HD95 - Mean (mm)     : {np.mean([m['mean_hd95'] for m in all_metrics]):.2f}")
    print(sep)
    print("  Glioma Grade Classification (High-Level vs Low-Level):")
    print(f"    Accuracy           : {clf_res['acc']*100:.2f} %")
    print(f"    F1 Score           : {clf_res['f1']:.4f}")
    print(f"    Precision          : {clf_res['precision']:.4f}")
    print(f"    Recall             : {clf_res['recall']:.4f}")
    print(f"    Confusion Matrix   : TP={clf_res['tp']}  FP={clf_res['fp']}  "
          f"FN={clf_res['fn']}  TN={clf_res['tn']}")
    print(sep)
    if args.save_predictions:
        print(f"  Predictions saved to : {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
