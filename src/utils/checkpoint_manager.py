"""
Checkpoint Manager.

Responsibilities
----------------
1. Save epoch / latest / best checkpoints with full metadata
   (epoch, loss, dice scores, split manifest path, data leakage status).
2. Restore model + optimizer + scheduler state for seamless resumption.
3. Guard-rail: refuses to save or load a checkpoint if the split manifest
   reports data leakage (calls SplitManager.assert_no_leakage()).
4. Maintains a rolling log of the last N checkpoints and auto-deletes older
   ones to keep disk usage bounded while always preserving latest and best.
5. Prints a rich status summary on every save / load.
"""

from __future__ import annotations
import os
import json
import glob
import shutil
import datetime
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CheckpointManager:
    """
    Unified checkpoint save / load for pre-training and fine-tuning stages.

    Parameters
    ----------
    checkpoint_dir  : Directory where checkpoints are written.
    stage           : 'pretrain' or 'finetune' — used in filename prefix.
    split_manager   : A SplitManager instance. Its assert_no_leakage()
                      is called before every save operation.
    keep_last_n     : Number of rolling epoch checkpoints to retain on
                      disk (best and latest checkpoints are always kept separately).
    metric_name     : Name of the primary metric used to determine "best".
    metric_mode     : 'max' → higher is better  |  'min' → lower is better.
    """

    LOG_FILE = "checkpoint_log.json"

    def __init__(
        self,
        checkpoint_dir: str,
        stage: str,
        split_manager,              # SplitManager instance
        keep_last_n: int = 5,
        metric_name: str = "val_loss",
        metric_mode: str = "min",
    ):
        assert stage in ("pretrain", "finetune"), \
            f"stage must be 'pretrain' or 'finetune', got '{stage}'"
        assert metric_mode in ("min", "max"), \
            f"metric_mode must be 'min' or 'max', got '{metric_mode}'"

        self.checkpoint_dir = checkpoint_dir
        self.stage          = stage
        self.sm             = split_manager
        self.keep_last_n    = keep_last_n
        self.metric_name    = metric_name
        self.metric_mode    = metric_mode

        os.makedirs(checkpoint_dir, exist_ok=True)

        self._log_path   = os.path.join(checkpoint_dir, self.LOG_FILE)
        self._best_path  = os.path.join(checkpoint_dir, f"{stage}_best.pt")
        self._latest_path= os.path.join(checkpoint_dir, f"{stage}_latest.pt")
        self._log: list[dict] = self._load_log()

        # Track best metric value
        if self._log:
            best_entries = [e for e in self._log if e.get("is_best")]
            if best_entries:
                self._best_metric = best_entries[-1]["metrics"].get(metric_name)
            else:
                self._best_metric = None
        else:
            self._best_metric = None

    # ── Public API ────────────────────────────────────────────────────────

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None,
        metrics: dict[str, float],
    ) -> str:
        """
        Saves a checkpoint for every epoch. Refuses if leakage is detected.
        Updates stage_latest.pt and stage_best.pt.
        """
        # Guard: abort on leakage
        self.sm.assert_no_leakage()

        primary = metrics.get(self.metric_name)
        is_best = self._is_better(primary)

        epoch_path = os.path.join(
            self.checkpoint_dir,
            f"{self.stage}_epoch_{epoch:04d}.pt"
        )

        payload = {
            "epoch":                epoch,
            "stage":                self.stage,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics":              metrics,
            "is_best":              is_best,
            "split_manifest_path":  self.sm._manifest_path,
            "leakage_detected":     False,
            "saved_at":             datetime.datetime.now().isoformat(),
        }
        
        # Save epoch checkpoint
        torch.save(payload, epoch_path)

        # Always save / overwrite latest checkpoint for easy resume
        torch.save(payload, self._latest_path)

        if is_best:
            if primary is not None:
                self._best_metric = primary
            shutil.copyfile(epoch_path, self._best_path)

        # Log entry
        log_entry = {
            "epoch":    epoch,
            "path":     epoch_path,
            "metrics":  metrics,
            "is_best":  is_best,
            "saved_at": payload["saved_at"],
        }
        self._log.append(log_entry)
        self._save_log()

        # Rolling cleanup
        self._cleanup_old_checkpoints()

        # Summary
        self._print_save_summary(epoch, epoch_path, metrics, is_best)
        return epoch_path

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        strict: bool = True,
    ) -> dict:
        """Loads the best checkpoint."""
        if not os.path.exists(self._best_path):
            raise FileNotFoundError(
                f"No best checkpoint found at {self._best_path}"
            )
        return self._load(self._best_path, model, optimizer, scheduler, strict)

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        strict: bool = True,
    ) -> dict | None:
        """Loads the most recent epoch/latest checkpoint. Returns None if none exist."""
        target_path = None
        if os.path.exists(self._latest_path):
            target_path = self._latest_path
        else:
            epoch_ckpts = sorted(
                glob.glob(os.path.join(self.checkpoint_dir, f"{self.stage}_epoch_*.pt"))
            )
            if epoch_ckpts:
                target_path = epoch_ckpts[-1]

        if not target_path:
            return None
        return self._load(target_path, model, optimizer, scheduler, strict)

    def load_from_path(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        strict: bool = True,
    ) -> dict:
        """Loads a specific checkpoint by file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return self._load(path, model, optimizer, scheduler, strict)

    def transfer_encoder(
        self,
        pretrained_path: str,
        finetune_model: nn.Module,
        device: torch.device,
        exclude_prefixes: tuple[str, ...] = ("recon_head.",),
    ) -> int:
        """
        Transfers encoder / decoder weights from pre-training checkpoint
        into fine-tuning model.
        """
        if not os.path.exists(pretrained_path):
            print(f"[CheckpointManager] WARNING: pretrained weights not found "
                  f"at {pretrained_path}. Fine-tuning from random init.")
            return 0

        ckpt = torch.load(pretrained_path, map_location=device)
        src_state = ckpt.get("model_state_dict", ckpt)
        tgt_state = finetune_model.state_dict()

        transferred = {}
        skipped_prefix = []
        shape_mismatch = []

        for k, v in src_state.items():
            if any(k.startswith(p) for p in exclude_prefixes):
                skipped_prefix.append(k)
                continue
            if k in tgt_state and tgt_state[k].shape == v.shape:
                transferred[k] = v
            elif k in tgt_state:
                shape_mismatch.append(k)

        tgt_state.update(transferred)
        finetune_model.load_state_dict(tgt_state, strict=False)

        print(
            f"[CheckpointManager] Encoder transfer from '{pretrained_path}':\n"
            f"  ✓ Transferred   : {len(transferred)}\n"
            f"  ✗ Skipped (head): {len(skipped_prefix)}\n"
            f"  ✗ Shape mismatch: {len(shape_mismatch)}"
        )
        return len(transferred)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _is_better(self, value: float | None) -> bool:
        if value is None:
            return False
        if self._best_metric is None:
            return True
        if self.metric_mode == "min":
            return value < self._best_metric
        return value > self._best_metric

    def _load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optimizer | None,
        scheduler: _LRScheduler | None,
        strict: bool,
    ) -> dict:
        device = next(model.parameters()).device
        payload = torch.load(path, map_location=device)

        if payload.get("leakage_detected", False):
            raise RuntimeError(
                f"[CheckpointManager] This checkpoint ({path}) was saved with "
                "data leakage detected. Discard it and retrain with a clean split."
            )

        model.load_state_dict(payload["model_state_dict"], strict=strict)

        if optimizer and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        epoch   = payload.get("epoch", 0)
        metrics = payload.get("metrics", {})
        print(
            f"[CheckpointManager] Loaded '{path}'\n"
            f"  Stage: {payload.get('stage','?')} | "
            f"Epoch: {epoch} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
        return payload

    def _load_log(self) -> list[dict]:
        if os.path.exists(self._log_path):
            with open(self._log_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return []

    def _save_log(self):
        with open(self._log_path, "w", encoding="utf-8") as fh:
            json.dump(self._log, fh, indent=2)

    def _cleanup_old_checkpoints(self):
        """Keep only the last `keep_last_n` epoch checkpoints on disk."""
        epoch_ckpts = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, f"{self.stage}_epoch_*.pt"))
        )
        to_delete = epoch_ckpts[: max(0, len(epoch_ckpts) - self.keep_last_n)]
        for path in to_delete:
            try:
                os.remove(path)
            except OSError:
                pass

    def _print_save_summary(
        self,
        epoch: int,
        path: str,
        metrics: dict[str, float],
        is_best: bool,
    ):
        best_marker = "  ★ NEW BEST" if is_best else ""
        metric_str  = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(
            f"[Checkpoint] Epoch {epoch:04d} saved → {os.path.basename(path)}"
            f"{best_marker}\n"
            f"             {metric_str}"
        )
