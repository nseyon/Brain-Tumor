"""
Pre-training Entrypoint: Self-Supervised 3D Intra-Modality Masked Image Modeling (IS-MIM)
for BraTS 2023 Adult Glioma Dataset.

Features:
- Deterministic 60/20/20 train/val/test data partition via SplitManager.
- Mixed precision AMP training for maximum speed.
- Per-epoch checkpoint saving with auto-resume capability.
- Clean tqdm progress bar showing step-by-step loss during training.
"""

from __future__ import annotations
import sys
import os
import argparse
import time
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
from src.losses import MaskedMSELoss
from src.utils import CheckpointManager


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="IS-MIM Self-Supervised Pre-training for BraTS 2023 Adult Glioma Dataset"
    )
    # Dataset
    p.add_argument("--data_dir",      type=str, default=r"C:\Glioma\Dataset\Glioma")
    p.add_argument("--manifest_dir",  type=str, default=r"C:\Glioma\checkpoints\split",
                   help="Where split_manifest.json / split_checklist.txt are stored")
    p.add_argument("--force_rebuild_split", action="store_true",
                   help="Force re-indexing and re-partitioning of subjects")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed for deterministic subject shuffle")
    p.add_argument("--no_checksums",  action="store_true",
                   help="Skip per-file SHA-256 checksums (faster)")

    # MIM parameters
    p.add_argument("--img_size",      type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--mim_patch_size",type=int, default=16)
    p.add_argument("--mask_ratio",    type=float, default=0.5)
    p.add_argument("--swap_prob",     type=float, default=0.5)

    # Training & Regularization
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--drop_rate",     type=float, default=0.1)
    p.add_argument("--drop_path_rate",type=float, default=0.1)

    # Hardware
    p.add_argument("--cpu",           action="store_true", help="Force CPU execution")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default=r"C:\Glioma\checkpoints\pretrain")
    p.add_argument("--save_interval",  type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--keep_last_n",    type=int, default=5, help="Rolling epoch checkpoints to keep on disk")
    p.add_argument("--resume",         action="store_true", help="Resume from latest checkpoint if available")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\n=== [IS-MIM Pre-training] Dataset: BraTS 2023 Adult Glioma | device={device} ===\n")

    # ── 1. Build / load the deterministic split ───────────────────────────
    sm = SplitManager(
        data_dir         = args.data_dir,
        manifest_dir     = args.manifest_dir,
        seed             = args.seed,
        train_ratio      = 0.60,
        val_ratio        = 0.20,
        force_rebuild    = args.force_rebuild_split,
        compute_checksums= not args.no_checksums,
    )
    train_ids, val_ids, test_ids = sm.get_splits()

    # ── 2. Build subject lists (train only) ───────────────────────────────
    train_subjects = [sm.get_subject(sid) for sid in train_ids]
    print(f"Pre-training on {len(train_subjects)} / {len(train_ids)+len(val_ids)+len(test_ids)} subjects (train split only)\n")

    # ── 3. Dataset & DataLoader ───────────────────────────────────────────
    train_ds = BraTS2023GliomaDataset(
        subjects      = train_subjects,
        mode          = "pretrain",
        target_shape  = tuple(args.img_size),
        patch_size    = args.mim_patch_size,
        mask_ratio    = args.mask_ratio,
        swap_prob     = args.swap_prob,
        augment       = False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    # ── 4. Model ──────────────────────────────────────────────────────────
    model = SwinMambaDual(
        in_chans       = 4,
        feat_size      = (48, 96, 192, 384),
        hidden_size    = 768,
        drop_rate      = args.drop_rate,
        drop_path_rate = args.drop_path_rate,
        mode           = "pretrain",
    ).to(device)

    criterion = MaskedMSELoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 5. Checkpoint manager ─────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        checkpoint_dir = args.checkpoint_dir,
        stage          = "pretrain",
        split_manager  = sm,
        keep_last_n    = args.keep_last_n,
        metric_name    = "train_loss",
        metric_mode    = "min",
    )

    start_epoch = 1
    best_loss   = float("inf")

    # Auto-resume check if flag set or checkpoint exists
    if args.resume or os.path.exists(os.path.join(args.checkpoint_dir, "pretrain_latest.pt")):
        payload = ckpt_mgr.load_latest(model, optimizer, scheduler)
        if payload:
            start_epoch = payload["epoch"] + 1
            if "metrics" in payload and "train_loss" in payload["metrics"]:
                best_loss = payload["metrics"]["train_loss"]
            print(f"Resumed pre-training from epoch {payload['epoch']} (start_epoch={start_epoch})\n")

    # ── 6. Training loop ──────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"Training for {args.epochs} epochs | steps/epoch={len(train_loader)}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}", leave=True)
        for batch in pbar:
            x_in   = batch["x_perturbed"].to(device, non_blocking=True)
            x_tgt  = batch["x_clean"].to(device, non_blocking=True)
            v_mask = batch["voxel_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    recon = model(x_in)
                    loss  = criterion(recon, x_tgt, voxel_mask=v_mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(x_in)
                loss  = criterion(recon, x_tgt, voxel_mask=v_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}", "avg_loss": f"{total_loss / (pbar.n + 1):.5f}"})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | avg_loss={avg_loss:.5f} | lr={lr_now:.2e} | time={elapsed:.1f}s")

        if epoch % args.save_interval == 0 or avg_loss < best_loss:
            ckpt_mgr.save(
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                metrics   = {"train_loss": avg_loss, "lr": lr_now},
            )
            best_loss = min(best_loss, avg_loss)

    print("\n=== IS-MIM Pre-training complete ===")


if __name__ == "__main__":
    main()
