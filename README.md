# 3D Brain Tumor Segmentation & Classification with SwinMamba and IS-MIM
### BraTS 2023 Adult Glioma Dataset

This repository implements an end-to-end medical vision pipeline that bridges:
1. **SwinMamba Backbone:** A hybrid 3D Transformer-SSM architecture combining local windowed self-attention (Swin Stage 0-1) and global tri-directional state-space modeling (Mamba Stage 2-3).
2. **Intra-Modality Masked Image Modeling (IS-MIM):** Self-supervised pretext pre-training via patch swapping across MRI modalities and 50% block token masking.
3. **Multi-Task Downstream Head:** Joint 3D dense segmentation (**Whole Tumor / WT**, **Tumor Core / TC**, **Enhancing Tumor / ET**) and **High-Level vs. Low-Level Glioma Classification** (HGG vs. LGG) on the **BraTS 2023 Adult Glioma** dataset.

---

## Directory Layout

```
C:\Glioma\
├── requirements.txt
├── README.md
├── pretrain.py                    # Self-supervised IS-MIM pre-training CLI
├── finetune.py                    # Multi-task segmentation & classification fine-tuning CLI
├── evaluate.py                    # Test set evaluation CLI
├── module_wise_output_demo.ipynb  # Interactive Jupyter Notebook for module-by-module walkthrough
├── src/
│   ├── models/
│   │   ├── selective_scan.py      # Unified selective scan (PyTorch fallback + CUDA hook)
│   │   ├── mamba_vision_mixer.py  # 3D Tri-Directional Mamba + GSC
│   │   ├── swin_blocks.py         # 3D Swin Transformer Block + 3D Window Attention
│   │   ├── unetr_blocks.py        # UNETR up-blocks + EV2 Residual Blocks
│   │   └── swin_mamba_dual.py     # Full SwinMambaDual architecture with dropout & task heads
│   ├── data/
│   │   ├── gbm_dataset.py         # BraTS 2023 Adult Glioma Dataset loader (.nii loading)
│   │   ├── split_manager.py       # Deterministic 60/20/20 train/val/test split manager
│   │   ├── is_mim_transforms.py   # 3D Patch swapping & block masking
│   │   └── transforms.py          # Intensity normalization, crop/pad, label mapping
│   ├── losses/
│   │   ├── mim_loss.py            # Masked MSE Loss
│   │   └── multitask_loss.py      # Dice + Focal (seg) + CrossEntropy (clf)
│   └── utils/
│       ├── metrics.py             # Volumetric Dice, HD95, Accuracy, F1
│       ├── checkpoint_manager.py  # Epoch & best checkpointing with auto-resume
│       └── postprocessing.py      # Sigmoid thresholding + cluster volume filtering
└── Dataset/
    └── Glioma/                    # Contains BraTS-GLI-*-* subject folders (.nii files)
```

---

## Architecture Overview

```
                         [ Stage 1: Self-Supervised Pre-training (IS-MIM) ]
                                                │
 4-Modality MRI Volumes                         ▼
 [T1n, T1c, T2w, T2f]  ──► [ 3D Stochastic Modality Swap + 50% Masking ] ──► [ SwinMamba Backbone ]
                                                                                      │
                                                                           [ 3x3x3 Recon Head ]
                                                                                      │
                                                                           MSE Loss on Masked Voxels
 ────────────────────────────────────────────────────────────────────────────────────────────────────
                         [ Stage 2: Downstream Multi-Task Fine-Tuning ]
                                                │
 Unmasked 4-Ch MRI       ──► [ SwinMamba 3D Encoder ]
                                    │               │
                            (Skip Connections)  (Bottleneck Features)
                                    │               │
                                    ▼               ▼
                        [ UNETR Decoder ]   [ GAP + GMP Pooling ]
                                    │               │
                        [ 1x1x1 Conv Head ] [ Dense MLP Head ]
                                    │               │
                               Segmentation    Classification
                              (WT, TC, ET)     (High vs Low Level)
```

---

## Setup & Installation

1. Activate your Python environment:
   ```bash
   conda activate mamba_env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Stage 1: Self-Supervised Pre-training (IS-MIM)
Pre-train the SwinMamba backbone on the 60% training split of the BraTS 2023 dataset:
```bash
python pretrain.py \
    --data_dir "C:\Glioma\Dataset\Glioma" \
    --batch_size 2 \
    --epochs 100 \
    --lr 5e-4 \
    --checkpoint_dir "checkpoints\pretrain"
```
*Note:* Automatically saves `pretrain_epoch_XXXX.pt` and `pretrain_latest.pt` after every epoch. If interrupted, simply append `--resume` to continue training from where it stopped.

---

### 2. Stage 2: Downstream Multi-Task Fine-Tuning
Transfer pre-trained weights and simultaneously train segmentation (WT, TC, ET) and classification:
```bash
python finetune.py \
    --data_dir "C:\Glioma\Dataset\Glioma" \
    --pretrained_weights "checkpoints\pretrain\pretrain_best.pt" \
    --batch_size 2 \
    --epochs 100 \
    --lr 2e-4 \
    --lambda_clf 0.5 \
    --checkpoint_dir "checkpoints\finetune"
```
*Outputs logged per epoch:*
- **Segmentation**: Mean Dice, WT Dice, TC Dice, ET Dice, HD95 (mm).
- **Classification**: Accuracy (%), F1 Score, Precision, Recall.
- Best checkpoint automatically updated as `finetune_best.pt`. Automatically saves latest epoch as `finetune_latest.pt` for seamless resume with `--resume`.

---

### 3. Stage 3: Test Set Evaluation & Prediction Export
Evaluate the best fine-tuned model against the locked 20% test split:
```bash
python evaluate.py \
    --data_dir "C:\Glioma\Dataset\Glioma" \
    --checkpoint_dir "checkpoints\finetune" \
    --save_predictions \
    --output_dir "outputs\predictions"
```

---

### 4. Interactive Module-Wise Output Walkthrough
To inspect shapes, value statistics (min, mean, max, std), and tensor transformations across every layer without cluttering training logs, open and run:
`module_wise_output_demo.ipynb`
