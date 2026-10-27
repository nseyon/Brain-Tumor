"""
Downstream Multi-Task Fine-Tuning for BraTS 2023 Adult Glioma Dataset:
Joint 3D Brain Tumor Segmentation (WT, TC, ET) and
Glioma Grade Classification (High-Level vs Low-Level).

Features:
- Deterministic 60/20/20 train/val/test split from SplitManager.
- Multi-task loss: Dice + Focal Loss (Segmentation) + CrossEntropy (Classification).
- Displays running step-by-step training progress bar with tqdm.
- Complete metric reporting on EVERY epoch (Dice WT, TC, ET, HD95, Acc, F1, Precision, Recall).
- Checkpoint saving on every epoch with auto-resume from best or latest checkpoint.
- Dropout & drop-path regularization to eliminate overfitting.
"""

from __future__ import annotations
import sys
import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.models import SwinMambaDual
from src.data import BraTS2023GliomaDataset, SplitManager
from src.losses import MultiTaskLoss
from src.utils import (
    CheckpointManager,
    evaluate_segmentation_case,
    compute_classification_metrics,
    postprocess_predictions,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Task Fine-Tuning: Segmentation + Classification for BraTS 2023"
    )
    p.add_argument("--data_dir",      type=str, default=r"C:\Glioma\Dataset\Glioma")
    p.add_argument("--manifest_dir",  type=str, default=r"C:\Glioma\checkpoints\split",
                   help="Directory of the split_manifest.json written by pretrain.py")
    p.add_argument("--no_checksums",  action="store_true")
    p.add_argument("--clinical_csv",  type=str, default=None,
                   help="Optional CSV with columns [subject_id, grade] for HGG/LGG labels")

    # Weights to transfer
    p.add_argument("--pretrained_weights", type=str,
                   default=r"C:\Glioma\checkpoints\pretrain\pretrain_best.pt",
                   help="IS-MIM pre-trained checkpoint to transfer encoder weights")

    p.add_argument("--img_size",      type=int, nargs=3, default=[96, 96, 96])

    # Training & Regularization
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--drop_rate",     type=float, default=0.1)
    p.add_argument("--drop_path_rate",type=float, default=0.1)
    p.add_argument("--lambda_clf",    type=float, default=0.5,
                   help="Weight on classification loss term")
    p.add_argument("--val_interval",  type=int,   default=1,
                   help="Run validation every N epochs")

    # Hardware
    p.add_argument("--cpu",           action="store_true", help="Force CPU execution")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default=r"C:\Glioma\checkpoints\finetune")
    p.add_argument("--keep_last_n",    type=int, default=5)
    p.add_argument("--resume",         action="store_true",
                   help="Resume from latest or best fine-tune epoch checkpoint")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_clinical_labels(csv_path: str | None) -> dict[str, int]:
    if csv_path is None:
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        id_col    = next(c for c in df.columns if c.lower() in
                         ["id", "subject_id", "patient_id", "brats_id", "bra_ts_id"])
        label_col = next(c for c in df.columns if c.lower() in
                         ["grade", "label", "hgg_lgg", "high_low"])
        labels = {}
        for _, row in df.iterrows():
            val = str(row[label_col]).lower()
            labels[str(row[id_col])] = (
                1 if any(h in val for h in ["high", "hgg", "4", "iv", "3", "iii"]) else 0
            )
        return labels
    except Exception as exc:
        print(f"[Warning] Could not load clinical CSV: {exc}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\n=== [Multi-Task Fine-Tuning] Dataset: BraTS 2023 Adult Glioma | device={device} ===\n")

    # ── 1. Load the split manifest ────────────────────────────────────────
    sm = SplitManager(
        data_dir          = args.data_dir,
        manifest_dir      = args.manifest_dir,
        force_rebuild     = False,
        compute_checksums = not args.no_checksums,
    )
    sm.assert_no_leakage()
    train_ids, val_ids, test_ids = sm.get_splits()

    print(f"Using {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test subjects\n")

    # ── 2. Build subject lists ────────────────────────────────────────────
    train_subjects = [sm.get_subject(sid) for sid in train_ids]
    val_subjects   = [sm.get_subject(sid) for sid in val_ids]

    clinical_labels = _load_clinical_labels(args.clinical_csv)

    # ── 3. Datasets & DataLoaders ─────────────────────────────────────────
    train_ds = BraTS2023GliomaDataset(
        subjects        = train_subjects,
        mode            = "finetune",
        target_shape    = tuple(args.img_size),
        augment         = True,
        clinical_labels = clinical_labels,
    )
    val_ds = BraTS2023GliomaDataset(
        subjects        = val_subjects,
        mode            = "finetune",
        target_shape    = tuple(args.img_size),
        augment         = False,
        clinical_labels = clinical_labels,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────
    model = SwinMambaDual(
        in_chans       = 4,
        seg_out_chans  = 3,
        num_classes    = 2,
        feat_size      = (48, 96, 192, 384),
        hidden_size    = 768,
        drop_rate      = args.drop_rate,
        drop_path_rate = args.drop_path_rate,
        mode           = "finetune",
    ).to(device)

    criterion = MultiTaskLoss(
        class_weights = [1.3, 1.0, 1.5],
        lambda_clf    = args.lambda_clf,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 5. Checkpoint manager ─────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        checkpoint_dir = args.checkpoint_dir,
        stage          = "finetune",
        split_manager  = sm,
        keep_last_n    = args.keep_last_n,
        metric_name    = "mean_dice",
        metric_mode    = "max",
    )

    start_epoch = 1
    scaler      = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Auto-resume or transfer pre-trained encoder weights
    if args.resume or os.path.exists(os.path.join(args.checkpoint_dir, "finetune_latest.pt")):
        payload = ckpt_mgr.load_latest(model, optimizer, scheduler)
        if payload:
            start_epoch = payload["epoch"] + 1
            print(f"Resumed fine-tuning from epoch {payload['epoch']} (start_epoch={start_epoch})\n")
    else:
        ckpt_mgr.transfer_encoder(
            pretrained_path = args.pretrained_weights,
            finetune_model  = model,
            device          = device,
            exclude_prefixes = ("recon_head.",),
        )

    # ── 6. Training + Validation loop ────────────────────────────────────
    print(f"\nFine-tuning for {args.epochs} epochs | train_steps={len(train_loader)} | val_steps={len(val_loader)}\n")

    for epoch in range(start_epoch, args.epochs + 1):

        # — Training Step —
        model.train()
        t_loss, t_seg, t_clf = 0.0, 0.0, 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} [Train]", leave=True)
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            tgts = {
                "seg":   batch["seg"].to(device, non_blocking=True),
                "label": batch["label"].to(device, non_blocking=True),
            }
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outs      = model(imgs)
                    loss_dict = criterion(outs, tgts)
                scaler.scale(loss_dict["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outs      = model(imgs)
                loss_dict = criterion(outs, tgts)
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            t_loss += loss_dict["loss"].item()
            t_seg  += loss_dict["loss_seg"].item()
            t_clf  += loss_dict["loss_clf"].item()

            pbar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "seg":  f"{loss_dict['loss_seg'].item():.4f}",
                "clf":  f"{loss_dict['loss_clf'].item():.4f}",
            })

        scheduler.step()
        n = len(train_loader)
        train_time = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs:03d} Train Summary | Total Loss={t_loss/n:.4f} (Seg={t_seg/n:.4f}, Clf={t_clf/n:.4f}) | {train_time:.1f}s")

        # — Validation Step —
        if epoch % args.val_interval == 0:
            model.eval()
            seg_metrics_list, clf_preds, clf_gts = [], [], []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    imgs     = batch["image"].to(device)
                    seg_gt   = batch["seg"].numpy()[0]
                    label_gt = batch["label"].item()

                    if scaler is not None:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outs = model(imgs)
                    else:
                        outs = model(imgs)

                    seg_raw  = outs["seg"][0].float().cpu().numpy()
                    clf_pred = torch.argmax(outs["clf"][0]).item()

                    clf_preds.append(clf_pred)
                    clf_gts.append(label_gt)

                    seg_bin = postprocess_predictions(seg_raw)
                    seg_metrics_list.append(
                        evaluate_segmentation_case(seg_bin, seg_gt)
                    )

            mean_dice = float(np.mean([m["mean_dice"] for m in seg_metrics_list]))
            mean_wt   = float(np.mean([m["dice_wt"]   for m in seg_metrics_list]))
            mean_tc   = float(np.mean([m["dice_tc"]   for m in seg_metrics_list]))
            mean_et   = float(np.mean([m["dice_et"]   for m in seg_metrics_list]))
            mean_hd95 = float(np.mean([m["mean_hd95"] for m in seg_metrics_list]))
            clf_res   = compute_classification_metrics(clf_preds, clf_gts)

            print(f"  [VAL METRICS] Dice Mean={mean_dice:.4f} (WT={mean_wt:.4f}, TC={mean_tc:.4f}, ET={mean_et:.4f}) | HD95={mean_hd95:.2f} mm")
            print(f"  [VAL CLASSIFICATION] Acc={clf_res['acc']*100:.2f}% | F1={clf_res['f1']:.4f} | Prec={clf_res['precision']:.4f} | Rec={clf_res['recall']:.4f}")

            # Save checkpoint every epoch + best model tracking
            ckpt_mgr.save(
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                metrics   = {
                    "mean_dice": mean_dice,
                    "dice_wt":   mean_wt,
                    "dice_tc":   mean_tc,
                    "dice_et":   mean_et,
                    "mean_hd95": mean_hd95,
                    "clf_acc":   clf_res["acc"],
                    "clf_f1":    clf_res["f1"],
                    "train_loss": t_loss / n,
                },
            )

    print("\n=== Multi-Task Fine-Tuning complete ===")


if __name__ == "__main__":
    main()
