"""
generate_architecture_png.py
─────────────────────────────
Renders a publication-grade, 300 DPI IEEE Transactions master system architecture diagram
('system_architecture.png') illustrating the complete end-to-end flow of the project.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_PNG = r"C:\Glioma\system_architecture.png"

def render_diagram():
    fig = plt.figure(figsize=(24, 14), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 1400)
    ax.axis("off")

    # Canvas Background
    bg = patches.Rectangle((0, 0), 2400, 1400, facecolor="#f8fafc", edgecolor="none")
    ax.add_patch(bg)
    canvas = patches.FancyBboxPatch((20, 20), 2360, 1360, boxstyle="round,pad=0,rounding_size=16",
                                    facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=2)
    ax.add_patch(canvas)

    # ─────────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────────
    ax.text(60, 1345, "End-to-End System Architecture: 3D SwinMamba Dual & IS-MIM Self-Supervised Learning",
            fontsize=22, fontweight="bold", color="#0f172a", va="top")
    ax.text(60, 1315, "Unified Deep Learning Pipeline for Multi-Parametric MRI Ingestion, 3D Intra-Modality Masked Pre-training, Joint Tumor Segmentation & Glioma Grade Classification",
            fontsize=12, color="#64748b", va="top")

    # Legend Badges
    legends = [
        ("Data Ingestion (mpMRI)", "#e0f2fe", "#0284c7", 1450),
        ("Hybrid 3D Encoder", "#f5f3ff", "#7c3aed", 1630),
        ("UNETR + EV2 Decoder", "#eff6ff", "#2563eb", 1810),
        ("Multi-Task Heads", "#fff7ed", "#ea580c", 1990),
        ("Inference & Post-Proc", "#ecfdf5", "#059669", 2170),
    ]
    for text, bg_col, border_col, x_pos in legends:
        box = patches.FancyBboxPatch((x_pos, 1315), 160, 32, boxstyle="round,pad=0,rounding_size=6",
                                     facecolor=bg_col, edgecolor=border_col, linewidth=1.5)
        ax.add_patch(box)
        circ = patches.Circle((x_pos + 14, 1331), 5, facecolor=border_col, edgecolor="none")
        ax.add_patch(circ)
        ax.text(x_pos + 26, 1331, text, fontsize=9.5, fontweight="bold", color="#1e293b", va="center")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: DATA INGESTION & DETERMINISTIC PARTITIONING (Left Column)
    # ─────────────────────────────────────────────────────────────────────────
    sec1 = patches.FancyBboxPatch((60, 40), 320, 1240, boxstyle="round,pad=0,rounding_size=10",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(sec1)
    h1 = patches.FancyBboxPatch((60, 1240), 320, 40, boxstyle="round,pad=0,rounding_size=10",
                                facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(h1)
    ax.text(75, 1260, "1. DATA INGESTION & SPLIT", fontsize=13, fontweight="bold", color="#1e293b", va="center")
    
    # 1a. Raw Modalities Box
    b1a = patches.FancyBboxPatch((75, 1020), 290, 200, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.5)
    ax.add_patch(b1a)
    ax.text(85, 1200, "Raw Multi-Parametric MRI (3D NIfTI)", fontsize=11, fontweight="bold", color="#0369a1")
    ax.text(85, 1180, "BraTS 2023 Cohort (1,251 Subjects)", fontsize=9.5, color="#64748b")
    
    modalities = [
        ("T1-Native (t1n.nii) — Anatomy", 1145),
        ("T1-Contrast (t1c.nii) — Gadolinium", 1115),
        ("T2-Weighted (t2w.nii) — Edema / Fluid", 1085),
        ("T2-FLAIR (t2f.nii) — Infiltration", 1055),
    ]
    for mod_text, y in modalities:
        m_box = patches.FancyBboxPatch((85, y), 270, 24, boxstyle="round,pad=0,rounding_size=4",
                                       facecolor="#f0f9ff", edgecolor="#bae6fd", linewidth=1)
        ax.add_patch(m_box)
        ax.text(95, y + 12, mod_text, fontsize=9, fontweight="bold", color="#0369a1", va="center")

    # Arrow down
    ax.annotate("", xy=(220, 970), xytext=(220, 1015),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    # 1b. Deterministic SplitManager Box
    b1b = patches.FancyBboxPatch((75, 760), 290, 205, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(b1b)
    ax.text(85, 945, "Deterministic SplitManager", fontsize=11, fontweight="bold", color="#0f172a")
    ax.text(85, 928, "Zero-Leakage Patient Partition (Seed=42)", fontsize=9, color="#64748b")

    splits = [
        ("Train Split: 750 (60.0%)", "Pre-train (750) & Finetune (750)", "#f8fafc", "#e2e8f0", "#1e293b", 870),
        ("Val Split: 250 (20.0%)", "Per-Epoch Val & Early Stopping", "#f8fafc", "#e2e8f0", "#1e293b", 820),
        ("Test Split: 251 (20.1%)", "Locked Unseen Test (evaluate.py)", "#fef2f2", "#fecaca", "#991b1b", 770),
    ]
    for title, desc, fcol, ecol, tcol, y in splits:
        sp_box = patches.FancyBboxPatch((85, y), 270, 42, boxstyle="round,pad=0,rounding_size=4",
                                        facecolor=fcol, edgecolor=ecol, linewidth=1)
        ax.add_patch(sp_box)
        ax.text(95, y + 28, title, fontsize=9.5, fontweight="bold", color=tcol)
        ax.text(95, y + 12, desc, fontsize=8.5, color="#64748b")

    # Arrow down
    ax.annotate("", xy=(220, 710), xytext=(220, 755),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    # 1c. Transforms & Normalization
    b1c = patches.FancyBboxPatch((75, 430), 290, 275, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#ffffff", edgecolor="#0d9488", linewidth=1.5)
    ax.add_patch(b1c)
    ax.text(85, 685, "Transforms & Normalization", fontsize=11, fontweight="bold", color="#0f766e")
    ax.text(85, 668, "src/data/transforms.py", fontsize=9, color="#64748b")

    preproc = [
        ("1. Foreground Robust Z-Score", "Clip [0.5%, 99.5%] | (x-μ)/(σ+1e-7)", 615),
        ("2. Spatial Standardization", "Center Crop/Pad → [4, 96, 96, 96]", 560),
        ("3. Multi-Region Label Map", "WT(1+2+3), TC(1+3), ET(3)", 505),
        ("4. Grade Label Extraction", "ET≥50 / TC≥150 → HGG(1) else LGG(0)", 450),
    ]
    for step_title, step_desc, y in preproc:
        p_box = patches.FancyBboxPatch((85, y), 270, 45, boxstyle="round,pad=0,rounding_size=4",
                                       facecolor="#f0fdfa", edgecolor="#ccfbf1", linewidth=1)
        ax.add_patch(p_box)
        ax.text(95, y + 30, step_title, fontsize=9.5, fontweight="bold", color="#0f766e")
        ax.text(95, y + 12, step_desc, fontsize=8.5, color="#115e59")

    # Arrow down
    ax.annotate("", xy=(220, 375), xytext=(220, 425),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    # 1d. Batch Ingestion Tensors Box
    b1d = patches.FancyBboxPatch((75, 70), 290, 300, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=1.5)
    ax.add_patch(b1d)
    ax.text(85, 345, "Batch Ingestion Tensors", fontsize=11, fontweight="bold", color="#0f172a")

    t1 = patches.FancyBboxPatch((85, 275), 270, 55, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1)
    ax.add_patch(t1)
    ax.text(95, 312, "Input Multi-Parametric MRI:", fontsize=9.5, fontweight="bold", color="#4338ca")
    ax.text(95, 290, "[B, 4, 96, 96, 96] (T1, T1c, T2, FLAIR)", fontsize=9, fontfamily="monospace", color="#1e293b")

    t2 = patches.FancyBboxPatch((85, 195), 270, 65, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1)
    ax.add_patch(t2)
    ax.text(95, 240, "Target Supervision:", fontsize=9.5, fontweight="bold", color="#059669")
    ax.text(95, 222, "Seg: [B, 3, 96, 96, 96] (WT, TC, ET)", fontsize=9, fontfamily="monospace", color="#1e293b")
    ax.text(95, 204, "Grade: [B] (0: LGG, 1: HGG)", fontsize=9, fontfamily="monospace", color="#1e293b")

    t3 = patches.FancyBboxPatch((85, 85), 270, 95, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1, linestyle="--")
    ax.add_patch(t3)
    ax.text(95, 160, "Online Data Augmentation:", fontsize=9, fontweight="bold", color="#475569")
    ax.text(95, 142, "• Random 3D Flips (X, Y, Z, p=0.5)", fontsize=8.5, color="#64748b")
    ax.text(95, 124, "• Random 90° Rotations (Axial)", fontsize=8.5, color="#64748b")
    ax.text(95, 106, "• Applied strictly to train split", fontsize=8.5, color="#64748b")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: IS-MIM PRETRAINING PIPELINE (Top Center)
    # ─────────────────────────────────────────────────────────────────────────
    sec2 = patches.FancyBboxPatch((410, 1080), 1380, 200, boxstyle="round,pad=0,rounding_size=10",
                                  facecolor="#faf5ff", edgecolor="#d8b4fe", linewidth=1.5)
    ax.add_patch(sec2)
    h2 = patches.FancyBboxPatch((410, 1240), 1380, 40, boxstyle="round,pad=0,rounding_size=10",
                                facecolor="#f3e8ff", edgecolor="#d8b4fe", linewidth=1.5)
    ax.add_patch(h2)
    ax.text(425, 1260, "2. SELF-SUPERVISED PRE-TRAINING PATH: INTRA-MODALITY MASKED IMAGE MODELING (IS-MIM)",
            fontsize=12, fontweight="bold", color="#6b21a8", va="center")

    steps_ismim = [
        ("2a. 3D Token Partition", "Patch size: 16³\n96³ → 6×6×6 grid\n216 Tokens / volume\n[B, 4, 16³, 216]", 430, 230),
        ("2b. Intra-Modality Swap", "Cross-modality swap p=0.5\n2 stochastic iterations\nT1 ↔ T1ce, T2 ↔ FLAIR\nLearns invariant anatomy", 690, 240),
        ("2c. 50% Block Masking", "Mask Ratio = 50%\n108 / 216 tokens zeroed\nForces spatial interpolation\nVoxel loss mask M [B, 1, 96³]", 960, 240),
        ("2d. Detokenization", "Reconstruct 3D volumes\nX_perturbed: [B, 4, 96³]\nX_clean: [B, 4, 96³]\nSent to SwinMamba Backbone", 1230, 240),
        ("2e. Reconstruction Loss", "L_MIM = Σ M⊙(X̂-X)² / Σ M\nEvaluated on corrupted voxels\nAdamW (lr=5e-4)\nCosine Annealing Schedule", 1500, 260),
    ]
    for title, desc, x, w in steps_ismim:
        sb = patches.FancyBboxPatch((x, 1095), w, 135, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor="#ffffff", edgecolor="#c084fc", linewidth=1.2)
        ax.add_patch(sb)
        sb_h = patches.FancyBboxPatch((x, 1205), w, 25, boxstyle="round,pad=0,rounding_size=6",
                                      facecolor="#f3e8ff", edgecolor="#c084fc", linewidth=1)
        ax.add_patch(sb_h)
        ax.text(x + 10, 1217, title, fontsize=9.5, fontweight="bold", color="#7e22ce", va="center")
        ax.text(x + 10, 1190, desc, fontsize=8.5, color="#334155", va="top")

    # Connectors between IS-MIM steps
    for x_arr in [665, 935, 1205, 1475]:
        ax.annotate("", xy=(x_arr + 20, 1162), xytext=(x_arr - 5, 1162),
                    arrowprops=dict(arrowstyle="->", color="#9333ea", lw=1.8))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: CORE ARCHITECTURE (3D SWINMAMBA DUAL + UNETR DECODER)
    # ─────────────────────────────────────────────────────────────────────────
    sec3 = patches.FancyBboxPatch((410, 40), 1380, 1010, boxstyle="round,pad=0,rounding_size=10",
                                  facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(sec3)
    h3 = patches.FancyBboxPatch((410, 1010), 1380, 40, boxstyle="round,pad=0,rounding_size=10",
                                facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(h3)
    ax.text(425, 1030, "3. UNIFIED HYBRID 3D SWINMAMBA DUAL BACKBONE & UNETR PROGRESSIVE DECODER",
            fontsize=13, fontweight="bold", color="#0f172a", va="center")
    ax.text(1050, 1030, "[Stages 0-1: Local Window Attention | Stages 2-3: Tri-Directional SSM | Decoder: UNETR + EV2]",
            fontsize=10.5, color="#64748b", va="center")

    # 3A. ENCODER SUB-PANEL (Left within Section 3)
    enc_panel = patches.FancyBboxPatch((425, 55), 520, 940, boxstyle="round,pad=0,rounding_size=8",
                                       facecolor="#f8fafc", edgecolor="#818cf8", linewidth=1.2)
    ax.add_patch(enc_panel)
    ax.text(440, 975, "3A. HYBRID 3D ENCODER (SwinTransformer + MambaVision)", fontsize=11, fontweight="bold", color="#3730a3")

    # Stem
    stem = patches.FancyBboxPatch((440, 900), 490, 55, boxstyle="round,pad=0,rounding_size=6",
                                  facecolor="#ffffff", edgecolor="#6366f1", linewidth=1.2)
    ax.add_patch(stem)
    ax.text(450, 938, "Encoder Stem Downsampler", fontsize=10, fontweight="bold", color="#312e81")
    ax.text(450, 915, "Conv3D (4→48, k=7³, s=2, p=3) + BN + LeakyReLU + Dropout3D", fontsize=8.5, color="#475569")
    ax.text(840, 925, "[B, 48, 48³]", fontsize=9, fontfamily="monospace", fontweight="bold", color="#4338ca")

    # Stage 0: Swin
    st0 = patches.FancyBboxPatch((440, 770), 490, 115, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#f5f3ff", edgecolor="#4f46e5", linewidth=1.2)
    ax.add_patch(st0)
    ax.text(450, 865, "STAGE 0: 3D Swin Transformer (Depth=2, Dims=48, Heads=3, Win=7³)", fontsize=9.5, fontweight="bold", color="#3730a3")
    ax.text(450, 845, "• Block 0: Window Attention (W-MSA) on [7, 7, 7] local cuboids", fontsize=8.5, color="#4c1d95")
    ax.text(450, 830, "• Block 1: Shifted Window Attention (SW-MSA, shift=(3, 3, 3))", fontsize=8.5, color="#4c1d95")
    ax.text(450, 810, "Norm + MlpChannel (48 → 96 → 48) + DropPath", fontsize=8.5, color="#64748b")
    ax.text(450, 785, "Output: [B, 48, 48, 48, 48]  ──→  Skip Connection S0", fontsize=9, fontweight="bold", color="#0284c7")

    # Patch Merge 0->1
    pm0 = patches.FancyBboxPatch((440, 725), 490, 32, boxstyle="round,pad=0,rounding_size=4",
                                 facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1)
    ax.add_patch(pm0)
    ax.text(450, 741, "Patch Merging 0→1: Conv3D (48→96, k=2³, s=2) → [B, 96, 24, 24, 24]", fontsize=8.5, color="#334155")

    # Stage 1: Swin
    st1 = patches.FancyBboxPatch((440, 595), 490, 115, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#f5f3ff", edgecolor="#4f46e5", linewidth=1.2)
    ax.add_patch(st1)
    ax.text(450, 690, "STAGE 1: 3D Swin Transformer (Depth=2, Dims=96, Heads=6, Win=7³)", fontsize=9.5, fontweight="bold", color="#3730a3")
    ax.text(450, 670, "• Block 0: Window Attention (W-MSA) on [7, 7, 7] local cuboids", fontsize=8.5, color="#4c1d95")
    ax.text(450, 655, "• Block 1: Shifted Window Attention (SW-MSA, shift=(3, 3, 3))", fontsize=8.5, color="#4c1d95")
    ax.text(450, 635, "Norm + MlpChannel (96 → 192 → 96) + DropPath", fontsize=8.5, color="#64748b")
    ax.text(450, 610, "Output: [B, 96, 24, 24, 24]  ──→  Skip Connection S1", fontsize=9, fontweight="bold", color="#0284c7")

    # Patch Merge 1->2
    pm1 = patches.FancyBboxPatch((440, 550), 490, 32, boxstyle="round,pad=0,rounding_size=4",
                                 facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1)
    ax.add_patch(pm1)
    ax.text(450, 566, "Patch Merging 1→2: Conv3D (96→192, k=2³, s=2) → [B, 192, 12, 12, 12]", fontsize=8.5, color="#334155")

    # Stage 2: Mamba + GSC
    st2 = patches.FancyBboxPatch((440, 375), 490, 160, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#faf5ff", edgecolor="#7c3aed", linewidth=1.2)
    ax.add_patch(st2)
    ax.text(450, 515, "STAGE 2: GSC + Tri-Directional Mamba SSM (Dims=192, Depth=2)", fontsize=9.5, fontweight="bold", color="#5b21b6")
    ax.text(450, 495, "• GSC Refinement: Residual Parallel 3×3×3 & 1×1×1 Convs + IN3D", fontsize=8.5, color="#6b21a8")
    ax.text(450, 475, "• Tri-Directional SSM Scans (Volumetric L = 12³ = 1,728 tokens):", fontsize=8.5, fontweight="bold", color="#4c1d95")
    ax.text(470, 458, "1. Axial Scan (D→H→W)   2. Sagittal Scan (W→D→H)   3. Coronal Scan (H→W→D)", fontsize=8, color="#5b21b6")
    ax.text(450, 435, "1D Depthwise Conv + Discrete State Space (d_state=16, expand=1)", fontsize=8.5, color="#64748b")
    ax.text(450, 415, "Norm + MlpChannel (192 → 384 → 192) + LayerScale(1e-6)", fontsize=8.5, color="#64748b")
    ax.text(450, 390, "Output: [B, 192, 12, 12, 12]  ──→  Skip Connection S2", fontsize=9, fontweight="bold", color="#0284c7")

    # Patch Merge 2->3
    pm2 = patches.FancyBboxPatch((440, 330), 490, 32, boxstyle="round,pad=0,rounding_size=4",
                                 facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1)
    ax.add_patch(pm2)
    ax.text(450, 346, "Patch Merging 2→3: Conv3D (192→384, k=2³, s=2) → [B, 384, 6, 6, 6]", fontsize=8.5, color="#334155")

    # Stage 3: Deepest Mamba
    st3 = patches.FancyBboxPatch((440, 70), 490, 245, boxstyle="round,pad=0,rounding_size=6",
                                 facecolor="#faf5ff", edgecolor="#7c3aed", linewidth=1.5)
    ax.add_patch(st3)
    ax.text(450, 295, "STAGE 3: Deepest Tri-Directional Mamba (Dims=384, Depth=2)", fontsize=10, fontweight="bold", color="#5b21b6")
    ax.text(450, 275, "• GSC Residual Spatial Priors at Bottleneck Scale", fontsize=8.5, color="#6b21a8")
    ax.text(450, 258, "• Orthogonal Volumetric Scans (Sequence L = 6³ = 216 tokens):", fontsize=8.5, fontweight="bold", color="#4c1d95")
    ax.text(470, 240, "Axial, Sagittal, Coronal Continuous State Modeling", fontsize=8, color="#5b21b6")
    ax.text(450, 218, "Linear Complexity O(N) captures whole-brain global spatial context", fontsize=8.5, color="#64748b")
    ax.text(450, 198, "Output Features: [B, 384, 6, 6, 6]  ──→  enc_bottleneck", fontsize=9, fontweight="bold", color="#1d4ed8")

    # 3B. SKIP ENCODERS COLUMN (Center Column within Section 3)
    skip_panel = patches.FancyBboxPatch((965, 55), 180, 940, boxstyle="round,pad=0,rounding_size=8",
                                        facecolor="#f0f9ff", edgecolor="#38bdf8", linewidth=1.2)
    ax.add_patch(skip_panel)
    ax.text(975, 975, "3B. SKIP ENCODERS", fontsize=10.5, fontweight="bold", color="#0369a1")

    skips = [
        ("enc_stem", "4 → 48", "[B, 48, 96³]", 890),
        ("enc_stage0", "48 → 48", "[B, 48, 48³]", 780),
        ("enc_stage1", "96 → 96", "[B, 96, 24³]", 620),
        ("enc_stage2", "192 → 192", "[B, 192, 12³]", 410),
    ]
    for name, dims, shape, y in skips:
        sk_b = patches.FancyBboxPatch((975, y), 160, 65, boxstyle="round,pad=0,rounding_size=6",
                                      facecolor="#ffffff", edgecolor="#0284c7", linewidth=1)
        ax.add_patch(sk_b)
        ax.text(985, y + 48, name, fontsize=9.5, fontweight="bold", color="#0284c7")
        ax.text(985, y + 30, dims, fontsize=8.5, color="#64748b")
        ax.text(985, y + 14, shape, fontsize=8.5, fontfamily="monospace", color="#0369a1")

    # Bottleneck Box
    btn_b = patches.FancyBboxPatch((975, 90), 160, 110, boxstyle="round,pad=0,rounding_size=6",
                                   facecolor="#eff6ff", edgecolor="#1d4ed8", linewidth=2)
    ax.add_patch(btn_b)
    ax.text(985, 175, "enc_bottleneck", fontsize=10, fontweight="bold", color="#1d4ed8")
    ax.text(985, 155, "384 → 768 dims", fontsize=9, color="#475569")
    ax.text(985, 135, "UnetrBasicBlock3D", fontsize=8.5, color="#64748b")
    ax.text(985, 110, "[B, 768, 6³]", fontsize=9.5, fontfamily="monospace", fontweight="bold", color="#1e40af")

    # 3C. UNETR PROGRESSIVE DECODER (Right within Section 3)
    dec_panel = patches.FancyBboxPatch((1165, 55), 610, 940, boxstyle="round,pad=0,rounding_size=8",
                                       facecolor="#f8fafc", edgecolor="#60a5fa", linewidth=1.2)
    ax.add_patch(dec_panel)
    ax.text(1180, 975, "3C. UNETR PROGRESSIVE DECODER + EV2 REFINEMENT", fontsize=11, fontweight="bold", color="#1e40af")

    # EV2 Final Refinement
    ref_b = patches.FancyBboxPatch((1180, 870), 580, 90, boxstyle="round,pad=0,rounding_size=6",
                                   facecolor="#ecfdf5", edgecolor="#059669", linewidth=2)
    ax.add_patch(ref_b)
    ax.text(1195, 938, "EV2 Residual Refinement Block (Final Latent Feature Map)", fontsize=10.5, fontweight="bold", color="#065f46")
    ax.text(1195, 918, "Channel Squeeze-and-Excitation: Y = φ(A(R(X))) + A(R(X))", fontsize=9, fontfamily="monospace", color="#047857")
    ax.text(1195, 898, "AdaptiveAvgPool3D → FC(48→12) → ReLU → FC(12→48) → Sigmoid Channel Weights", fontsize=8, color="#64748b")
    ax.text(1195, 880, "Final Latent Output: dec_out [B, 48, 96, 96, 96]", fontsize=9.5, fontweight="bold", color="#059669")

    # Decoders 1 to 4
    decoders = [
        ("Decoder 1 (UnetrUpBlock3D): 48 → 48 (48³ → 96³)",
         "ConvTranspose3D (48→48, k=2, s=2) ⊕ Concat with skip_stem (48)",
         "EV2Block Fusion (96→48) → Output: [B, 48, 96, 96, 96]", 705),
        ("Decoder 2 (UnetrUpBlock3D): 96 → 48 (24³ → 48³)",
         "ConvTranspose3D (96→48, k=2, s=2) ⊕ Concat with skip_0 (48)",
         "EV2Block Fusion (96→48) → Output: [B, 48, 48, 48, 48]", 525),
        ("Decoder 3 (UnetrUpBlock3D): 192 → 96 (12³ → 24³)",
         "ConvTranspose3D (192→96, k=2, s=2) ⊕ Concat with skip_1 (96)",
         "EV2Block Fusion (192→96) → Output: [B, 96, 24, 24, 24]", 325),
        ("Decoder 4 (UnetrUpBlock3D): 768 → 192 (6³ → 12³)",
         "ConvTranspose3D (768→192, k=2, s=2) ⊕ Concat with skip_2 (192)",
         "EV2Block Fusion (384→192) → Output: [B, 192, 12, 12, 12]", 110),
    ]
    for title, l1, l2, y in decoders:
        d_box = patches.FancyBboxPatch((1180, y), 580, 110, boxstyle="round,pad=0,rounding_size=6",
                                       facecolor="#ffffff", edgecolor="#3b82f6", linewidth=1.2)
        ax.add_patch(d_box)
        ax.text(1195, y + 88, title, fontsize=9.5, fontweight="bold", color="#1d4ed8")
        ax.text(1195, y + 65, l1, fontsize=8.5, color="#334155")
        ax.text(1195, y + 42, l2, fontsize=8.5, color="#166534")

    # Vertical decoder arrows (4 -> 3 -> 2 -> 1 -> Refine)
    for y_arr in [230, 445, 645, 825]:
        ax.annotate("", xy=(1470, y_arr + 85), xytext=(1470, y_arr),
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2))

    # Horizontal skip connections from Center Column into Decoders
    skip_routes = [
        (965 + 160, 920, 1180, 760), # skip_stem -> dec1
        (965 + 160, 810, 1180, 580), # skip_0 -> dec2
        (965 + 160, 650, 1180, 380), # skip_1 -> dec3
        (965 + 160, 440, 1180, 165), # skip_2 -> dec4
        (965 + 160, 145, 1180, 130), # bottleneck -> dec4
    ]
    for x1, y1, x2, y2 in skip_routes:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.5, linestyle="--"))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: TASK HEADS, POST-PROCESSING & CLINICAL METRICS (Right Column)
    # ─────────────────────────────────────────────────────────────────────────
    sec4 = patches.FancyBboxPatch((1810, 40), 530, 1240, boxstyle="round,pad=0,rounding_size=10",
                                  facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(sec4)
    h4 = patches.FancyBboxPatch((1810, 1240), 530, 40, boxstyle="round,pad=0,rounding_size=10",
                                facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(h4)
    ax.text(1825, 1260, "4. TASK HEADS, LOSS & TEST INFERENCE", fontsize=13, fontweight="bold", color="#1e293b", va="center")

    # 4A. Segmentation Head
    sh_b = patches.FancyBboxPatch((1825, 990), 500, 230, boxstyle="round,pad=0,rounding_size=6",
                                  facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.5)
    ax.add_patch(sh_b)
    ax.text(1840, 1200, "HEAD 1: 3D Dense Tumor Segmentation", fontsize=11, fontweight="bold", color="#0369a1")
    ax.text(1840, 1180, "UnetOutBlock3D: Conv3D (48 → 3, k=1³, s=1, bias=True)", fontsize=9, color="#475569")
    
    seg_sub = patches.FancyBboxPatch((1840, 1070), 470, 95, boxstyle="round,pad=0,rounding_size=4",
                                     facecolor="#f0f9ff", edgecolor="#bae6fd", linewidth=1)
    ax.add_patch(seg_sub)
    ax.text(1850, 1148, "Output Channels (Multi-Label Sigmoid Activation):", fontsize=9, fontweight="bold", color="#0284c7")
    ax.text(1850, 1130, "• Channel 0: Tumor Core (TC) = NCR + ET", fontsize=8.5, color="#0369a1")
    ax.text(1850, 1113, "• Channel 1: Whole Tumor (WT) = NCR + ED + ET", fontsize=8.5, color="#0369a1")
    ax.text(1850, 1096, "• Channel 2: Enhancing Tumor (ET) = Active Contrast Core", fontsize=8.5, color="#0369a1")
    ax.text(1850, 1078, "Raw Logits: [B, 3, 96, 96, 96] → Sigmoid Probabilities", fontsize=8.5, fontfamily="monospace", color="#0369a1")

    ax.text(1840, 1050, "Segmentation Loss Objective:", fontsize=9, fontweight="bold", color="#c2410c")
    ax.text(1840, 1032, "L_seg = SoftDiceLoss(TC=1.3, WT=1.0, ET=1.5) + SigmoidFocalLoss(α=0.25, γ=2.0)", fontsize=8, color="#9a3412")

    # 4B. Classification Head
    ch_b = patches.FancyBboxPatch((1825, 730), 500, 240, boxstyle="round,pad=0,rounding_size=6",
                                  facecolor="#ffffff", edgecolor="#7c3aed", linewidth=1.5)
    ax.add_patch(ch_b)
    ax.text(1840, 950, "HEAD 2: Glioma Grade Classification", fontsize=11, fontweight="bold", color="#5b21b6")
    ax.text(1840, 930, "Input: Bottleneck Latent Representation [B, 768, 6, 6, 6]", fontsize=9, color="#475569")

    clf_sub = patches.FancyBboxPatch((1840, 830), 470, 90, boxstyle="round,pad=0,rounding_size=4",
                                     facecolor="#faf5ff", edgecolor="#d8b4fe", linewidth=1)
    ax.add_patch(clf_sub)
    ax.text(1850, 905, "Dual Global Pooling & Feature Concatenation:", fontsize=9, fontweight="bold", color="#6b21a8")
    ax.text(1850, 888, "• Global Avg Pooling (GAP): [B, 768]  +  Global Max Pooling (GMP): [B, 768]", fontsize=8, color="#4c1d95")
    ax.text(1850, 870, "• Fused 1536-dim Representation: [GAP ‖ GMP]", fontsize=8.5, fontfamily="monospace", color="#4c1d95")
    ax.text(1850, 852, "• MLP: 1536 → 256 (Drop 0.4) → 128 (Drop 0.2) → 2 Classes", fontsize=8.5, color="#5b21b6")
    ax.text(1850, 835, "• Output Logits: [B, 2] → Softmax [P(LGG), P(HGG)]", fontsize=8.5, fontfamily="monospace", color="#5b21b6")

    ax.text(1840, 805, "Classification Loss Objective:", fontsize=9, fontweight="bold", color="#c2410c")
    ax.text(1840, 788, "L_clf = CrossEntropyLoss(label_smoothing=0.05)", fontsize=8.5, color="#9a3412")
    ax.text(1840, 768, "Total Multi-Task Loss: L_total = L_seg + 0.5 · L_clf", fontsize=9.5, fontweight="bold", color="#c2410c")

    # 4C. Post-Processing & CCA
    pp_b = patches.FancyBboxPatch((1825, 430), 500, 280, boxstyle="round,pad=0,rounding_size=6",
                                  facecolor="#ffffff", edgecolor="#059669", linewidth=1.5)
    ax.add_patch(pp_b)
    ax.text(1840, 690, "5. Clinical Post-Processing Pipeline", fontsize=11, fontweight="bold", color="#065f46")
    ax.text(1840, 672, "src/utils/postprocessing.py", fontsize=9, color="#64748b")

    pp_steps = [
        ("1. Sigmoid Thresholding:", "Binary mask extraction: p(voxel) > 0.5 → {0, 1}", 620),
        ("2. 3D Connected Component Analysis:", "Removes false-positive small clusters (scipy.ndimage.label):", 565),
    ]
    for t_step, d_step, y in pp_steps:
        s_box = patches.FancyBboxPatch((1840, y), 470, 42, boxstyle="round,pad=0,rounding_size=4",
                                       facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1)
        ax.add_patch(s_box)
        ax.text(1850, y + 26, t_step, fontsize=9, fontweight="bold", color="#15803d")
        ax.text(1850, y + 10, d_step, fontsize=8, color="#166534")

    cca_box = patches.FancyBboxPatch((1840, 450), 470, 95, boxstyle="round,pad=0,rounding_size=4",
                                     facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1)
    ax.add_patch(cca_box)
    ax.text(1850, 528, "Anatomical Volumetric Thresholds:", fontsize=9, fontweight="bold", color="#047857")
    ax.text(1850, 508, "• Whole Tumor (WT) cluster < 250 voxels → Set to 0", fontsize=8.5, color="#14532d")
    ax.text(1850, 490, "• Tumor Core (TC) cluster < 150 voxels → Set to 0", fontsize=8.5, color="#14532d")
    ax.text(1850, 472, "• Enhancing Tumor (ET) cluster < 100 voxels → Set to 0", fontsize=8.5, color="#14532d")

    # 4D. Evaluation & Deliverables
    eval_b = patches.FancyBboxPatch((1825, 70), 500, 340, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor="#ffffff", edgecolor="#10b981", linewidth=1.5)
    ax.add_patch(eval_b)
    ax.text(1840, 390, "Quantitative Evaluation & Final Deliverables", fontsize=11, fontweight="bold", color="#065f46")
    ax.text(1840, 372, "Held-Out Unseen Test Cohort: 251 Patients (evaluate.py)", fontsize=9, color="#64748b")

    out_metrics = [
        ("Dense 3D Segmentation Metrics:", "Dice WT, TC, ET, Mean Dice | 95th Percentile Hausdorff Distance (HD95 mm) | RVD | Sensitivity | Specificity", 305),
        ("Glioma Grade Classification Metrics:", "Accuracy (%) | F1-Score | Precision | Recall | AUC-ROC | Confusion Matrix", 240),
        ("Exportable System Deliverables:", "• Reconstructed 3D NIfTI Masks (outputs/predictions/*.nii.gz)\n• 10 Master Evaluation Graphs (outputs/graphs/)\n• Interactive Clinical MRI Web App (app.py)\n• Module-Wise Tensor Verification (demo_pipeline.py)", 95),
    ]
    for m_title, m_desc, y in out_metrics:
        h = 50 if "\n" not in m_desc else 125
        m_box = patches.FancyBboxPatch((1840, y), 470, h, boxstyle="round,pad=0,rounding_size=4",
                                       facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1)
        ax.add_patch(m_box)
        ax.text(1850, y + h - 16, m_title, fontsize=9, fontweight="bold", color="#1e293b")
        ax.text(1850, y + h - 34, m_desc, fontsize=8, color="#475569", va="top")

    # Final Connectors from Dec_out and Bottleneck into Right Column
    ax.annotate("", xy=(1825, 1110), xytext=(1760, 915),
                arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2))
    ax.annotate("", xy=(1825, 850), xytext=(1135, 145),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2, linestyle="--"))

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated 300 DPI PNG diagram at: {OUTPUT_PNG}")

if __name__ == "__main__":
    render_diagram()
