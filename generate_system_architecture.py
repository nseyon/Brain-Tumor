"""
generate_system_architecture.py
───────────────────────────────
Generates IEEE-publication-quality system architecture diagrams:
1. 'system_architecture.svg'  - Scalable vector graphics (SVG) with publication typography,
                                crisp vector lines, exact tensor shapes, and component layouts.
2. 'system_architecture.drawio' - Native Draw.io / diagrams.net XML file with interactive,
                                fully editable nodes, containers, and connector routes.
Strictly based on the actual implementation in c:\\Glioma.
"""

import os
import xml.etree.ElementTree as ET

WORKSPACE = r"C:\Glioma"
SVG_PATH = os.path.join(WORKSPACE, "system_architecture.svg")
DRAWIO_PATH = os.path.join(WORKSPACE, "system_architecture.drawio")

def create_svg():
    # Overall canvas dimensions (2400 x 1400 px, high resolution for IEEE Transactions 2-column or double-page spread)
    W, H = 2400, 1420
    
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif">
  <defs>
    <!-- Gradients -->
    <linearGradient id="grad-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f8fafc" />
      <stop offset="100%" stop-color="#f1f5f9" />
    </linearGradient>
    <linearGradient id="grad-input" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#0369a1" />
    </linearGradient>
    <linearGradient id="grad-preproc" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0d9488" />
      <stop offset="100%" stop-color="#0f766e" />
    </linearGradient>
    <linearGradient id="grad-swin" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4f46e5" />
      <stop offset="100%" stop-color="#4338ca" />
    </linearGradient>
    <linearGradient id="grad-mamba" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7c3aed" />
      <stop offset="100%" stop-color="#6d28d9" />
    </linearGradient>
    <linearGradient id="grad-unetr" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#1d4ed8" />
    </linearGradient>
    <linearGradient id="grad-ev2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#059669" />
      <stop offset="100%" stop-color="#047857" />
    </linearGradient>
    <linearGradient id="grad-loss" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ea580c" />
      <stop offset="100%" stop-color="#c2410c" />
    </linearGradient>
    <linearGradient id="grad-post" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#dc2626" />
      <stop offset="100%" stop-color="#b91c1c" />
    </linearGradient>

    <!-- Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed" />
    </marker>
    <marker id="arrow-teal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0d9488" />
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#ea580c" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
    </marker>

    <!-- Shadows -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <filter id="card-shadow" x="-3%" y="-3%" width="106%" height="110%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.05"/>
    </filter>
  </defs>

  <!-- Background Canvas -->
  <rect width="{W}" height="{H}" fill="url(#grad-bg)" />
  <rect x="20" y="20" width="{W-40}" height="{H-40}" rx="16" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" filter="url(#shadow)" />

  <!-- HEADER -->
  <g transform="translate(60, 45)">
    <text x="0" y="32" font-size="28" font-weight="800" fill="#0f172a" letter-spacing="-0.5">End-to-End System Architecture: 3D SwinMamba Dual &amp; IS-MIM Self-Supervised Learning</text>
    <text x="0" y="58" font-size="15" font-weight="500" fill="#64748b">Unified Deep Learning Pipeline for Multi-Parametric MRI Ingestion, 3D Intra-Modality Masked Pre-training, Joint Tumor Segmentation &amp; Glioma Grade Classification</text>
    
    <!-- Legend / Badges in Header -->
    <g transform="translate(1420, 10)">
      <rect x="0" y="0" width="160" height="32" rx="6" fill="#e0f2fe" stroke="#38bdf8" stroke-width="1.5" />
      <circle cx="16" cy="16" r="6" fill="#0284c7" />
      <text x="32" y="21" font-size="12" font-weight="700" fill="#0369a1">Data Ingestion (mpMRI)</text>

      <rect x="175" y="0" width="160" height="32" rx="6" fill="#f5f3ff" stroke="#a78bfa" stroke-width="1.5" />
      <circle cx="191" cy="16" r="6" fill="#7c3aed" />
      <text x="207" y="21" font-size="12" font-weight="700" fill="#6d28d9">Hybrid 3D Encoder</text>

      <rect x="350" y="0" width="160" height="32" rx="6" fill="#eff6ff" stroke="#60a5fa" stroke-width="1.5" />
      <circle cx="366" cy="16" r="6" fill="#2563eb" />
      <text x="382" y="21" font-size="12" font-weight="700" fill="#1d4ed8">UNETR + EV2 Decoder</text>

      <rect x="525" y="0" width="160" height="32" rx="6" fill="#fff7ed" stroke="#fb923c" stroke-width="1.5" />
      <circle cx="541" cy="16" r="6" fill="#ea580c" />
      <text x="557" y="21" font-size="12" font-weight="700" fill="#c2410c">Multi-Task Heads</text>

      <rect x="700" y="0" width="145" height="32" rx="6" fill="#ecfdf5" stroke="#34d399" stroke-width="1.5" />
      <circle cx="716" cy="16" r="6" fill="#059669" />
      <text x="732" y="21" font-size="12" font-weight="700" fill="#047857">Inference &amp; Post-Proc</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SECTION 1: DATA INGESTION & DETERMINISTIC PARTITIONING (Left Column)      -->
  <!-- ========================================================================= -->
  <g transform="translate(60, 130)">
    <!-- Container Box -->
    <rect x="0" y="0" width="340" height="1220" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" filter="url(#card-shadow)" />
    <rect x="0" y="0" width="340" height="42" rx="12" fill="#f1f5f9" />
    <path d="M 0 30 L 0 42 L 340 42 L 340 30 Z" fill="#f1f5f9" />
    <text x="16" y="26" font-size="15" font-weight="700" fill="#1e293b">1. DATA INGESTION &amp; SPLIT</text>
    <rect x="250" y="10" width="75" height="22" rx="4" fill="#0284c7" />
    <text x="258" y="25" font-size="11" font-weight="700" fill="#ffffff">BraTS 2023</text>

    <!-- Raw MRI Modalities -->
    <g transform="translate(16, 56)">
      <rect x="0" y="0" width="308" height="190" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
      <text x="14" y="24" font-size="13" font-weight="700" fill="#0f172a">Raw Multi-Parametric MRI (3D NIfTI)</text>
      <text x="14" y="42" font-size="11" fill="#64748b">1,251 Adult Glioma Subjects | Axial, Sag, Cor</text>

      <!-- 4 Channels list -->
      <g transform="translate(14, 52)">
        <rect x="0" y="0" width="280" height="26" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1" />
        <text x="10" y="17" font-size="11" font-weight="600" fill="#0369a1">T1-Native (t1n.nii) — Non-Contrast T1 Anatomy</text>

        <rect x="0" y="32" width="280" height="26" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1" />
        <text x="10" y="49" font-size="11" font-weight="600" fill="#0369a1">T1-Contrast (t1c.nii) — Gadolinium Hyperintensity</text>

        <rect x="0" y="64" width="280" height="26" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1" />
        <text x="10" y="81" font-size="11" font-weight="600" fill="#0369a1">T2-Weighted (t2w.nii) — Hyperintense Edema/Fluid</text>

        <rect x="0" y="96" width="280" height="26" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1" />
        <text x="10" y="113" font-size="11" font-weight="600" fill="#0369a1">T2-FLAIR (t2f.nii) — Peritumoral Infiltration</text>
      </g>
    </g>

    <!-- Flow Arrow -->
    <line x1="170" y1="248" x2="170" y2="276" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

    <!-- Split Manager Box -->
    <g transform="translate(16, 276)">
      <rect x="0" y="0" width="308" height="185" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
      <text x="14" y="24" font-size="13" font-weight="700" fill="#0f172a">Deterministic SplitManager</text>
      <text x="14" y="40" font-size="11" fill="#64748b">Zero-Leakage Patient Partitioning (Seed=42)</text>
      
      <!-- Sub-splits -->
      <g transform="translate(14, 52)">
        <rect x="0" y="0" width="280" height="34" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
        <text x="10" y="18" font-size="11" font-weight="700" fill="#1e293b">Train Split (60.0%): 750 Subjects</text>
        <text x="10" y="30" font-size="10" fill="#64748b">IS-MIM Pre-train (750) &amp; Finetune (750)</text>

        <rect x="0" y="40" width="280" height="34" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
        <text x="10" y="58" font-size="11" font-weight="700" fill="#1e293b">Val Split (20.0%): 250 Subjects</text>
        <text x="10" y="70" font-size="10" fill="#64748b">Per-Epoch Validation &amp; Early Stopping</text>

        <rect x="0" y="80" width="280" height="34" rx="4" fill="#fef2f2" stroke="#fecaca" stroke-width="1" />
        <text x="10" y="98" font-size="11" font-weight="700" fill="#991b1b">Test Split (20.1%): 251 Subjects</text>
        <text x="10" y="110" font-size="10" fill="#dc2626">Locked Unseen Test Split (evaluate.py)</text>
      </g>
    </g>

    <!-- Flow Arrow -->
    <line x1="170" y1="463" x2="170" y2="491" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

    <!-- Standardization & Preprocessing -->
    <g transform="translate(16, 491)">
      <rect x="0" y="0" width="308" height="235" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
      <text x="14" y="24" font-size="13" font-weight="700" fill="#0f172a">Preprocessing &amp; Normalization</text>
      <text x="14" y="40" font-size="11" fill="#64748b">src/data/transforms.py Pipeline</text>

      <g transform="translate(14, 52)">
        <rect x="0" y="0" width="280" height="42" rx="4" fill="#f0fdfa" stroke="#ccfbf1" stroke-width="1" />
        <text x="10" y="18" font-size="11" font-weight="700" fill="#0f766e">1. Foreground Robust Z-Score</text>
        <text x="10" y="32" font-size="10" fill="#115e59">Clip [0.5%, 99.5%] | (x - μ) / (σ + 1e-7)</text>

        <rect x="0" y="48" width="280" height="42" rx="4" fill="#f0fdfa" stroke="#ccfbf1" stroke-width="1" />
        <text x="10" y="66" font-size="11" font-weight="700" fill="#0f766e">2. Spatial Standardization</text>
        <text x="10" y="80" font-size="10" fill="#115e59">Isotropic Center Crop/Pad → [4, 96, 96, 96]</text>

        <rect x="0" y="96" width="280" height="42" rx="4" fill="#f0fdfa" stroke="#ccfbf1" stroke-width="1" />
        <text x="10" y="114" font-size="11" font-weight="700" fill="#0f766e">3. Multi-Region Label Mapping</text>
        <text x="10" y="128" font-size="10" fill="#115e59">Raw Seg → WT (1+2+3), TC (1+3), ET (3)</text>

        <rect x="0" y="144" width="280" height="32" rx="4" fill="#f0fdfa" stroke="#ccfbf1" stroke-width="1" />
        <text x="10" y="160" font-size="11" font-weight="700" fill="#0f766e">4. Grade Label Extraction</text>
        <text x="10" y="171" font-size="10" fill="#115e59">ET ≥ 50 / TC ≥ 150 voxels → HGG (1), else LGG (0)</text>
      </g>
    </g>

    <!-- Flow Arrow -->
    <line x1="170" y1="728" x2="170" y2="756" stroke="#475569" stroke-width="2" marker-end="url(#arrow)" />

    <!-- Output Tensor of Pipeline -->
    <g transform="translate(16, 756)">
      <rect x="0" y="0" width="308" height="150" rx="8" fill="#f1f5f9" stroke="#94a3b8" stroke-width="2" />
      <text x="14" y="24" font-size="13" font-weight="700" fill="#0f172a">Batch Ingestion Tensor Formats</text>
      
      <rect x="14" y="38" width="280" height="44" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
      <text x="22" y="56" font-size="11" font-weight="700" fill="#4338ca">Finetune / Inference Volume:</text>
      <text x="22" y="72" font-size="11" font-family="'Consolas', monospace" fill="#1e293b">[B, 4, 96, 96, 96] (T1, T1c, T2, FLAIR)</text>

      <rect x="14" y="90" width="280" height="46" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
      <text x="22" y="108" font-size="11" font-weight="700" fill="#059669">Target Ground Truths:</text>
      <text x="22" y="124" font-size="11" font-family="'Consolas', monospace" fill="#1e293b">Seg: [B, 3, 96, 96, 96] | Clf: [B] (0/1)</text>
    </g>

    <!-- Augmentation note -->
    <g transform="translate(16, 918)">
      <rect x="0" y="0" width="308" height="70" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3" />
      <text x="12" y="20" font-size="11" font-weight="700" fill="#475569">Online Data Augmentation (Train Mode):</text>
      <text x="12" y="38" font-size="10" fill="#64748b">• Random 3D Flips along X, Y, Z axes (p=0.5)</text>
      <text x="12" y="54" font-size="10" fill="#64748b">• Random 90° Rotations in Axial plane (k∈{1,2,3})</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SECTION 2: INTRA-MODALITY MASKED IMAGE MODELING (IS-MIM) (Top Center)     -->
  <!-- ========================================================================= -->
  <g transform="translate(420, 130)">
    <rect x="0" y="0" width="1380" height="230" rx="12" fill="#faf5ff" stroke="#d8b4fe" stroke-width="2" filter="url(#card-shadow)" />
    <rect x="0" y="0" width="1380" height="38" rx="12" fill="#f3e8ff" />
    <path d="M 0 25 L 0 38 L 1380 38 L 1380 25 Z" fill="#f3e8ff" />
    <text x="16" y="24" font-size="14" font-weight="700" fill="#6b21a8">2. SELF-SUPERVISED PRE-TRAINING PATH: INTRA-MODALITY MASKED IMAGE MODELING (IS-MIM)</text>
    <rect x="1240" y="8" width="125" height="22" rx="4" fill="#9333ea" />
    <text x="1248" y="23" font-size="11" font-weight="700" fill="#ffffff">Pre-train: 10 Epochs</text>

    <!-- Sub-steps in horizontal layout -->
    <g transform="translate(20, 50)">
      <!-- 2a. 3D Patch Partitioning -->
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="240" height="150" rx="8" fill="#ffffff" stroke="#c084fc" stroke-width="1.5" />
        <rect x="0" y="0" width="240" height="28" rx="8" fill="#f3e8ff" />
        <text x="10" y="19" font-size="12" font-weight="700" fill="#7e22ce">2a. 3D Token Partition</text>
        <text x="10" y="48" font-size="11" fill="#475569">Patch size: <tspan font-weight="700">16 × 16 × 16</tspan></text>
        <text x="10" y="66" font-size="11" fill="#475569">Volume: 96³ → 6×6×6 grid</text>
        <text x="10" y="88" font-size="11" font-weight="700" fill="#6b21a8">Tokens: 216 cuboids</text>
        <rect x="10" y="102" width="220" height="34" rx="4" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1" />
        <text x="16" y="124" font-size="10" font-family="'Consolas', monospace" fill="#581c87">[B, 4, 16³, 216 tokens]</text>
      </g>

      <!-- Arrow -->
      <line x1="242" y1="75" x2="275" y2="75" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)" />

      <!-- 2b. Intra-Modality Swap -->
      <g transform="translate(278, 0)">
        <rect x="0" y="0" width="265" height="150" rx="8" fill="#ffffff" stroke="#c084fc" stroke-width="1.5" />
        <rect x="0" y="0" width="265" height="28" rx="8" fill="#f3e8ff" />
        <text x="10" y="19" font-size="12" font-weight="700" fill="#7e22ce">2b. Intra-Modality Swap</text>
        <text x="10" y="48" font-size="11" fill="#475569">Cross-modality swap prob: <tspan font-weight="700">p = 0.5</tspan></text>
        <text x="10" y="66" font-size="11" fill="#475569">Stochastic 2-iteration pair exchange</text>
        <text x="10" y="84" font-size="10" fill="#64748b">T1 ↔ T1ce, T2 ↔ FLAIR pair permutations</text>
        <rect x="10" y="102" width="245" height="34" rx="4" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1" />
        <text x="16" y="124" font-size="10" font-weight="700" fill="#581c87">Encodes modality-invariant anatomy</text>
      </g>

      <!-- Arrow -->
      <line x1="545" y1="75" x2="578" y2="75" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)" />

      <!-- 2c. 50% Block Masking -->
      <g transform="translate(581, 0)">
        <rect x="0" y="0" width="250" height="150" rx="8" fill="#ffffff" stroke="#c084fc" stroke-width="1.5" />
        <rect x="0" y="0" width="250" height="28" rx="8" fill="#f3e8ff" />
        <text x="10" y="19" font-size="12" font-weight="700" fill="#7e22ce">2c. 50% Block-Wise Masking</text>
        <text x="10" y="48" font-size="11" fill="#475569">Mask ratio: <tspan font-weight="700">50% (108/216)</tspan></text>
        <text x="10" y="66" font-size="11" fill="#475569">Selected cuboids set to 0.0</text>
        <text x="10" y="84" font-size="10" fill="#64748b">Generates binary voxel_mask M</text>
        <rect x="10" y="102" width="230" height="34" rx="4" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1" />
        <text x="16" y="124" font-size="10" font-weight="700" fill="#581c87">Forces spatial context synthesis</text>
      </g>

      <!-- Arrow -->
      <line x1="833" y1="75" x2="866" y2="75" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)" />

      <!-- 2d. Detokenize to Perturbed Volume -->
      <g transform="translate(869, 0)">
        <rect x="0" y="0" width="230" height="150" rx="8" fill="#ffffff" stroke="#c084fc" stroke-width="1.5" />
        <rect x="0" y="0" width="230" height="28" rx="8" fill="#f3e8ff" />
        <text x="10" y="19" font-size="12" font-weight="700" fill="#7e22ce">2d. Detokenization</text>
        <text x="10" y="48" font-size="11" fill="#475569">Tokens → 3D Volume</text>
        <text x="10" y="66" font-size="11" font-weight="700" fill="#1e293b">X_perturbed: [B, 4, 96, 96, 96]</text>
        <text x="10" y="84" font-size="11" font-weight="700" fill="#1e293b">X_clean: [B, 4, 96, 96, 96]</text>
        <rect x="10" y="102" width="210" height="34" rx="4" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1" />
        <text x="16" y="124" font-size="10" font-family="'Consolas', monospace" fill="#581c87">Voxel mask: [B, 1, 96³, 1]</text>
      </g>

      <!-- Arrow -->
      <line x1="1101" y1="75" x2="1134" y2="75" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)" />

      <!-- 2e. Pre-train Objective -->
      <g transform="translate(1137, 0)">
        <rect x="0" y="0" width="195" height="150" rx="8" fill="#ffffff" stroke="#ea580c" stroke-width="1.5" />
        <rect x="0" y="0" width="195" height="28" rx="8" fill="#ffedd5" />
        <text x="10" y="19" font-size="12" font-weight="700" fill="#c2410c">Pre-train Loss (Masked MSE)</text>
        <text x="10" y="48" font-size="10" fill="#475569">L_MIM = Σ M ⊙ (X̂ - X)² / Σ M</text>
        <text x="10" y="68" font-size="10" fill="#475569">Evaluated on corrupted</text>
        <text x="10" y="82" font-size="10" fill="#475569">voxels only (3x3 conv head)</text>
        <rect x="10" y="102" width="175" height="34" rx="4" fill="#fff7ed" stroke="#fed7aa" stroke-width="1" />
        <text x="14" y="118" font-size="10" font-weight="700" fill="#9a3412">AdamW (lr=5e-4)</text>
        <text x="14" y="130" font-size="9" fill="#c2410c">CosineAnnealingLR</text>
      </g>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SECTION 3: CORE MODEL ARCHITECTURE: 3D SWINMAMBA DUAL (Center Area)        -->
  <!-- ========================================================================= -->
  <g transform="translate(420, 385)">
    <!-- Main Model Container -->
    <rect x="0" y="0" width="1380" height="965" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="2" filter="url(#card-shadow)" />
    <rect x="0" y="0" width="1380" height="42" rx="12" fill="#f8fafc" />
    <path d="M 0 30 L 0 42 L 1380 42 L 1380 30 Z" fill="#f8fafc" />
    <text x="20" y="27" font-size="16" font-weight="800" fill="#0f172a">3. UNIFIED HYBRID 3D SWINMAMBA DUAL BACKBONE &amp; UNETR DECODER</text>
    <text x="650" y="27" font-size="13" font-weight="600" fill="#475569">[Stages 0-1: 3D Swin Local Attention | Stages 2-3: Tri-Directional Mamba SSM | Decoder: UNETR + EV2]</text>

    <!-- LEFT COLUMN OF MODEL: HYBRID ENCODER -->
    <g transform="translate(25, 60)">
      <!-- Encoder Group Box -->
      <rect x="0" y="0" width="560" height="880" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5" />
      <rect x="0" y="0" width="560" height="36" rx="10" fill="#ede9fe" />
      <text x="16" y="24" font-size="14" font-weight="700" fill="#5b21b6">HYBRID 3D ENCODER (SwinTransformer + MambaVision)</text>

      <!-- Input Stem -->
      <g transform="translate(20, 50)">
        <rect x="0" y="0" width="520" height="54" rx="6" fill="#ffffff" stroke="#818cf8" stroke-width="1.5" />
        <text x="14" y="22" font-size="12" font-weight="700" fill="#312e81">Encoder Stem Downsampler</text>
        <text x="14" y="40" font-size="11" fill="#475569">Conv3D (4 → 48, k=7³, s=2, p=3) + BatchNorm3D + LeakyReLU(0.01) + Dropout3D(0.05)</text>
        <rect x="400" y="10" width="110" height="34" rx="4" fill="#e0e7ff" />
        <text x="408" y="32" font-size="11" font-family="'Consolas', monospace" font-weight="700" fill="#3730a3">[B, 48, 48³]</text>
      </g>

      <!-- Flow Arrow -->
      <line x1="280" y1="104" x2="280" y2="124" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- STAGE 0: SWIN 3D -->
      <g transform="translate(20, 124)">
        <rect x="0" y="0" width="520" height="120" rx="6" fill="#ffffff" stroke="#6366f1" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="26" rx="6" fill="#e0e7ff" />
        <text x="12" y="18" font-size="12" font-weight="700" fill="#3730a3">STAGE 0: 3D Swin Transformer (Depth=2, Dims=48, Heads=3, Win=7³)</text>
        <g transform="translate(14, 34)">
          <rect x="0" y="0" width="240" height="42" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
          <text x="10" y="18" font-size="11" font-weight="700" fill="#4c1d95">Block 0: 3D W-MSA</text>
          <text x="10" y="32" font-size="10" fill="#6d28d9">Window Attention [7, 7, 7] | Shift: (0,0,0)</text>

          <rect x="250" y="0" width="240" height="42" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
          <text x="10" y="18" font-size="11" font-weight="700" fill="#4c1d95">Block 1: 3D SW-MSA</text>
          <text x="10" y="32" font-size="10" fill="#6d28d9">Shifted Window [7, 7, 7] | Shift: (3,3,3)</text>
        </g>
        <text x="14" y="102" font-size="10" fill="#475569">Norm + MlpChannel (48 → 96 → 48)  |  Output: <tspan font-family="'Consolas', monospace" font-weight="700" fill="#4338ca">[B, 48, 48, 48, 48]</tspan> → <tspan font-weight="700" fill="#0284c7">Skip S0</tspan></text>
      </g>

      <!-- Patch Merging 0 -> 1 -->
      <g transform="translate(20, 252)">
        <rect x="0" y="0" width="520" height="32" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" />
        <text x="14" y="20" font-size="11" font-weight="600" fill="#334155">Patch Merging 0→1: BatchNorm3D + Conv3D (48 → 96, k=2³, s=2) → [B, 96, 24, 24, 24]</text>
      </g>

      <!-- Flow Arrow -->
      <line x1="280" y1="284" x2="280" y2="304" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- STAGE 1: SWIN 3D -->
      <g transform="translate(20, 304)">
        <rect x="0" y="0" width="520" height="120" rx="6" fill="#ffffff" stroke="#6366f1" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="26" rx="6" fill="#e0e7ff" />
        <text x="12" y="18" font-size="12" font-weight="700" fill="#3730a3">STAGE 1: 3D Swin Transformer (Depth=2, Dims=96, Heads=6, Win=7³)</text>
        <g transform="translate(14, 34)">
          <rect x="0" y="0" width="240" height="42" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
          <text x="10" y="18" font-size="11" font-weight="700" fill="#4c1d95">Block 0: 3D W-MSA</text>
          <text x="10" y="32" font-size="10" fill="#6d28d9">Window Attention [7, 7, 7] | Shift: (0,0,0)</text>

          <rect x="250" y="0" width="240" height="42" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
          <text x="10" y="18" font-size="11" font-weight="700" fill="#4c1d95">Block 1: 3D SW-MSA</text>
          <text x="10" y="32" font-size="10" fill="#6d28d9">Shifted Window [7, 7, 7] | Shift: (3,3,3)</text>
        </g>
        <text x="14" y="102" font-size="10" fill="#475569">Norm + MlpChannel (96 → 192 → 96)  |  Output: <tspan font-family="'Consolas', monospace" font-weight="700" fill="#4338ca">[B, 96, 24, 24, 24]</tspan> → <tspan font-weight="700" fill="#0284c7">Skip S1</tspan></text>
      </g>

      <!-- Patch Merging 1 -> 2 -->
      <g transform="translate(20, 432)">
        <rect x="0" y="0" width="520" height="32" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" />
        <text x="14" y="20" font-size="11" font-weight="600" fill="#334155">Patch Merging 1→2: BatchNorm3D + Conv3D (96 → 192, k=2³, s=2) → [B, 192, 12, 12, 12]</text>
      </g>

      <!-- Flow Arrow -->
      <line x1="280" y1="464" x2="280" y2="484" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- STAGE 2: GSC + TRI-DIRECTIONAL MAMBA -->
      <g transform="translate(20, 484)">
        <rect x="0" y="0" width="520" height="160" rx="6" fill="#ffffff" stroke="#8b5cf6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="26" rx="6" fill="#ede9fe" />
        <text x="12" y="18" font-size="12" font-weight="700" fill="#5b21b6">STAGE 2: GSC + Tri-Directional Mamba SSM (Dims=192, Depth=2, L=1,728)</text>
        
        <!-- GSC box -->
        <rect x="14" y="34" width="492" height="34" rx="4" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1" />
        <text x="24" y="48" font-size="11" font-weight="700" fill="#6b21a8">GSC (Global Spatial Context Refinement):</text>
        <text x="24" y="61" font-size="10" fill="#475569">Parallel (3×3×3 Conv) &amp; (1×1×1 Conv) with InstanceNorm3D + ReLU + Residual Connection</text>

        <!-- Tri-directional Mamba box -->
        <g transform="translate(14, 74)">
          <rect x="0" y="0" width="492" height="60" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
          <text x="10" y="16" font-size="11" font-weight="700" fill="#4c1d95">Tri-Directional Selective State-Space Scans (Axial, Sagittal, Coronal):</text>
          
          <rect x="10" y="24" width="145" height="28" rx="3" fill="#ffffff" stroke="#a78bfa" stroke-width="1" />
          <text x="16" y="38" font-size="10" font-weight="700" fill="#5b21b6">Axial Scan (D→H→W)</text>
          <text x="16" y="48" font-size="9" fill="#6d28d9">Depthwise 1D Conv + SSM</text>

          <rect x="165" y="24" width="155" height="28" rx="3" fill="#ffffff" stroke="#a78bfa" stroke-width="1" />
          <text x="171" y="38" font-size="10" font-weight="700" fill="#5b21b6">Sagittal Scan (W→D→H)</text>
          <text x="171" y="48" font-size="9" fill="#6d28d9">Depthwise 1D Conv + SSM</text>

          <rect x="330" y="24" width="150" height="28" rx="3" fill="#ffffff" stroke="#a78bfa" stroke-width="1" />
          <text x="336" y="38" font-size="10" font-weight="700" fill="#5b21b6">Coronal Scan (H→W→D)</text>
          <text x="336" y="48" font-size="9" fill="#6d28d9">Depthwise 1D Conv + SSM</text>
        </g>
        <text x="14" y="148" font-size="10" fill="#475569">MlpChannel (192 → 384 → 192)  |  Output: <tspan font-family="'Consolas', monospace" font-weight="700" fill="#5b21b6">[B, 192, 12, 12, 12]</tspan> → <tspan font-weight="700" fill="#0284c7">Skip S2</tspan></text>
      </g>

      <!-- Patch Merging 2 -> 3 -->
      <g transform="translate(20, 652)">
        <rect x="0" y="0" width="520" height="32" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" />
        <text x="14" y="20" font-size="11" font-weight="600" fill="#334155">Patch Merging 2→3: BatchNorm3D + Conv3D (192 → 384, k=2³, s=2) → [B, 384, 6, 6, 6]</text>
      </g>

      <!-- Flow Arrow -->
      <line x1="280" y1="684" x2="280" y2="704" stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- STAGE 3: GSC + TRI-DIRECTIONAL MAMBA (Bottleneck Level) -->
      <g transform="translate(20, 704)">
        <rect x="0" y="0" width="520" height="150" rx="6" fill="#ffffff" stroke="#8b5cf6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="26" rx="6" fill="#ede9fe" />
        <text x="12" y="18" font-size="12" font-weight="700" fill="#5b21b6">STAGE 3: Deepest Mamba SSM (Dims=384, Depth=2, Sequence L=216)</text>
        <text x="14" y="48" font-size="11" fill="#475569">GSC Block (Residual 3×3×3 &amp; 1×1×1 convolutions) + LayerScale (1e-6)</text>
        <text x="14" y="66" font-size="11" fill="#475569">Orthogonal Volumetric SSM State Size: <tspan font-weight="700">d_state=16, d_conv=3, expand=1</tspan></text>
        <text x="14" y="86" font-size="10" fill="#64748b">Captures long-range continuous 3D context across entire patient brain space</text>
        
        <rect x="14" y="98" width="492" height="38" rx="4" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1" />
        <text x="24" y="116" font-size="11" font-weight="700" fill="#5b21b6">Deepest Encoder Features:</text>
        <text x="24" y="129" font-size="11" font-family="'Consolas', monospace" fill="#1e293b">[B, 384, 6, 6, 6] → Passed to enc_bottleneck</text>
      </g>
    </g>

    <!-- CENTER COLUMN: SKIP ENCODER CONNECTIONS (Vertical Highway) -->
    <g transform="translate(605, 60)">
      <rect x="0" y="0" width="170" height="880" rx="10" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.5" />
      <rect x="0" y="0" width="170" height="36" rx="10" fill="#e0f2fe" />
      <text x="12" y="24" font-size="13" font-weight="700" fill="#0369a1">SKIP ENCODERS</text>

      <!-- Skip Stem -->
      <g transform="translate(10, 52)">
        <rect x="0" y="0" width="150" height="52" rx="6" fill="#ffffff" stroke="#38bdf8" stroke-width="1.5" />
        <text x="8" y="18" font-size="11" font-weight="700" fill="#0284c7">enc_stem (4→48)</text>
        <text x="8" y="32" font-size="9" fill="#64748b">UnetrBasicBlock3D</text>
        <text x="8" y="44" font-size="9" font-family="'Consolas', monospace" font-weight="700" fill="#0369a1">[B, 48, 96³]</text>
      </g>

      <!-- Skip 0 -->
      <g transform="translate(10, 158)">
        <rect x="0" y="0" width="150" height="52" rx="6" fill="#ffffff" stroke="#38bdf8" stroke-width="1.5" />
        <text x="8" y="18" font-size="11" font-weight="700" fill="#0284c7">enc_stage0 (48→48)</text>
        <text x="8" y="32" font-size="9" fill="#64748b">UnetrBasicBlock3D</text>
        <text x="8" y="44" font-size="9" font-family="'Consolas', monospace" font-weight="700" fill="#0369a1">[B, 48, 48³]</text>
      </g>

      <!-- Skip 1 -->
      <g transform="translate(10, 338)">
        <rect x="0" y="0" width="150" height="52" rx="6" fill="#ffffff" stroke="#38bdf8" stroke-width="1.5" />
        <text x="8" y="18" font-size="11" font-weight="700" fill="#0284c7">enc_stage1 (96→96)</text>
        <text x="8" y="32" font-size="9" fill="#64748b">UnetrBasicBlock3D</text>
        <text x="8" y="44" font-size="9" font-family="'Consolas', monospace" font-weight="700" fill="#0369a1">[B, 96, 24³]</text>
      </g>

      <!-- Skip 2 -->
      <g transform="translate(10, 538)">
        <rect x="0" y="0" width="150" height="52" rx="6" fill="#ffffff" stroke="#38bdf8" stroke-width="1.5" />
        <text x="8" y="18" font-size="11" font-weight="700" fill="#0284c7">enc_stage2 (192→192)</text>
        <text x="8" y="32" font-size="9" fill="#64748b">UnetrBasicBlock3D</text>
        <text x="8" y="44" font-size="9" font-family="'Consolas', monospace" font-weight="700" fill="#0369a1">[B, 192, 12³]</text>
      </g>

      <!-- Bottleneck Block -->
      <g transform="translate(10, 755)">
        <rect x="0" y="0" width="150" height="70" rx="6" fill="#ffffff" stroke="#2563eb" stroke-width="2" />
        <text x="8" y="18" font-size="11" font-weight="700" fill="#1d4ed8">enc_bottleneck</text>
        <text x="8" y="32" font-size="10" fill="#475569">384 → 768 dims</text>
        <rect x="6" y="42" width="138" height="22" rx="3" fill="#dbeafe" />
        <text x="10" y="57" font-size="10" font-family="'Consolas', monospace" font-weight="700" fill="#1e40af">[B, 768, 6, 6, 6]</text>
      </g>
    </g>

    <!-- Skip Connection Routing Lines (Curved/Orthogonal paths) -->
    <!-- From raw input / stem to enc_stem -->
    <path d="M -40 76 L 605 76" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="4,3" fill="none" />
    <!-- From stage 0 to enc_stage0 -->
    <path d="M 545 220 L 605 184" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="4,3" fill="none" marker-end="url(#arrow-blue)" />
    <!-- From stage 1 to enc_stage1 -->
    <path d="M 545 400 L 605 364" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="4,3" fill="none" marker-end="url(#arrow-blue)" />
    <!-- From stage 2 to enc_stage2 -->
    <path d="M 545 600 L 605 564" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="4,3" fill="none" marker-end="url(#arrow-blue)" />
    <!-- From stage 3 to enc_bottleneck -->
    <path d="M 545 830 L 605 800" stroke="#1d4ed8" stroke-width="2" fill="none" marker-end="url(#arrow-blue)" />

    <!-- RIGHT COLUMN OF MODEL: PROGRESSIVE UNETR + EV2 DECODER -->
    <g transform="translate(795, 60)">
      <rect x="0" y="0" width="560" height="880" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5" />
      <rect x="0" y="0" width="560" height="36" rx="10" fill="#dbeafe" />
      <text x="16" y="24" font-size="14" font-weight="700" fill="#1e40af">PROGRESSIVE UNETR DECODER + EV2 RESIDUAL REFINEMENT</text>

      <!-- Decoder 4 (Bottleneck -> Level 3) -->
      <g transform="translate(20, 725)">
        <rect x="0" y="0" width="520" height="98" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="24" rx="6" fill="#eff6ff" />
        <text x="12" y="17" font-size="12" font-weight="700" fill="#1d4ed8">Decoder 4 (UnetrUpBlock3D): 768 → 192 (6³ → 12³)</text>
        <text x="12" y="42" font-size="11" fill="#475569">ConvTranspose3d (768 → 192, k=2, s=2) ⊕ Concat with <tspan font-weight="700" fill="#0284c7">skip_2 (192)</tspan></text>
        <rect x="12" y="52" width="496" height="34" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="20" y="68" font-size="10" font-weight="700" fill="#166534">EV2Block Fusion (384 → 192): Squeeze-and-Excitation Channel Attention + Residual Conv</text>
        <text x="20" y="80" font-size="10" font-family="'Consolas', monospace" fill="#14532d">Output: [B, 192, 12, 12, 12]</text>
      </g>

      <!-- Up Arrow 4 to 3 -->
      <line x1="280" y1="725" x2="280" y2="695" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />

      <!-- Decoder 3 (Level 3 -> Level 2) -->
      <g transform="translate(20, 560)">
        <rect x="0" y="0" width="520" height="98" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="24" rx="6" fill="#eff6ff" />
        <text x="12" y="17" font-size="12" font-weight="700" fill="#1d4ed8">Decoder 3 (UnetrUpBlock3D): 192 → 96 (12³ → 24³)</text>
        <text x="12" y="42" font-size="11" fill="#475569">ConvTranspose3d (192 → 96, k=2, s=2) ⊕ Concat with <tspan font-weight="700" fill="#0284c7">skip_1 (96)</tspan></text>
        <rect x="12" y="52" width="496" height="34" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="20" y="68" font-size="10" font-weight="700" fill="#166534">EV2Block Fusion (192 → 96): Squeeze-and-Excitation Channel Attention + Residual Conv</text>
        <text x="20" y="80" font-size="10" font-family="'Consolas', monospace" fill="#14532d">Output: [B, 96, 24, 24, 24]</text>
      </g>

      <!-- Up Arrow 3 to 2 -->
      <line x1="280" y1="560" x2="280" y2="530" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />

      <!-- Decoder 2 (Level 2 -> Level 1) -->
      <g transform="translate(20, 395)">
        <rect x="0" y="0" width="520" height="98" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="24" rx="6" fill="#eff6ff" />
        <text x="12" y="17" font-size="12" font-weight="700" fill="#1d4ed8">Decoder 2 (UnetrUpBlock3D): 96 → 48 (24³ → 48³)</text>
        <text x="12" y="42" font-size="11" fill="#475569">ConvTranspose3d (96 → 48, k=2, s=2) ⊕ Concat with <tspan font-weight="700" fill="#0284c7">skip_0 (48)</tspan></text>
        <rect x="12" y="52" width="496" height="34" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="20" y="68" font-size="10" font-weight="700" fill="#166534">EV2Block Fusion (96 → 48): Squeeze-and-Excitation Channel Attention + Residual Conv</text>
        <text x="20" y="80" font-size="10" font-family="'Consolas', monospace" fill="#14532d">Output: [B, 48, 48, 48, 48]</text>
      </g>

      <!-- Up Arrow 2 to 1 -->
      <line x1="280" y1="395" x2="280" y2="365" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />

      <!-- Decoder 1 (Level 1 -> Full Res) -->
      <g transform="translate(20, 230)">
        <rect x="0" y="0" width="520" height="98" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5" />
        <rect x="0" y="0" width="520" height="24" rx="6" fill="#eff6ff" />
        <text x="12" y="17" font-size="12" font-weight="700" fill="#1d4ed8">Decoder 1 (UnetrUpBlock3D): 48 → 48 (48³ → 96³)</text>
        <text x="12" y="42" font-size="11" fill="#475569">ConvTranspose3d (48 → 48, k=2, s=2) ⊕ Concat with <tspan font-weight="700" fill="#0284c7">skip_stem (48)</tspan></text>
        <rect x="12" y="52" width="496" height="34" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="20" y="68" font-size="10" font-weight="700" fill="#166534">EV2Block Fusion (96 → 48): Squeeze-and-Excitation Channel Attention + Residual Conv</text>
        <text x="20" y="80" font-size="10" font-family="'Consolas', monospace" fill="#14532d">Output: [B, 48, 96, 96, 96]</text>
      </g>

      <!-- Up Arrow 1 to Refinement -->
      <line x1="280" y1="230" x2="280" y2="200" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />

      <!-- EV2 Refinement Block (Final) -->
      <g transform="translate(20, 65)">
        <rect x="0" y="0" width="520" height="110" rx="8" fill="#ffffff" stroke="#059669" stroke-width="2" />
        <rect x="0" y="0" width="520" height="28" rx="8" fill="#d1fae5" />
        <text x="14" y="19" font-size="12" font-weight="700" fill="#065f46">EV2 Residual Refinement Block (Final Decoder Feature Map)</text>
        <text x="14" y="46" font-size="11" fill="#334155">Channel Squeeze-and-Excitation: <tspan font-family="'Consolas', monospace">Y = φ(A(R(X))) + A(R(X))</tspan></text>
        <text x="14" y="64" font-size="10" fill="#64748b">AdaptiveAvgPool3D → FC(48→12) → ReLU → FC(12→48) → Sigmoid Channel Weights</text>
        <rect x="14" y="74" width="492" height="28" rx="4" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
        <text x="24" y="92" font-size="11" font-weight="700" fill="#047857">Final Refined Decoder Latent: <tspan font-family="'Consolas', monospace">[B, 48, 96, 96, 96]</tspan></text>
      </g>

      <!-- Skip connections from Center Column into Decoders -->
      <!-- Skip stem to Dec1 -->
      <path d="M -20 110 L 20 250" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-blue)" />
      <!-- Skip 0 to Dec2 -->
      <path d="M -20 216 L 20 415" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-blue)" />
      <!-- Skip 1 to Dec3 -->
      <path d="M -20 396 L 20 580" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-blue)" />
      <!-- Skip 2 to Dec4 -->
      <path d="M -20 596 L 20 745" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="3,3" fill="none" marker-end="url(#arrow-blue)" />
      <!-- Bottleneck to Dec4 -->
      <path d="M -20 825 L 20 780" stroke="#1d4ed8" stroke-width="2" fill="none" marker-end="url(#arrow-blue)" />
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SECTION 4: TASK HEADS & LOSS FORMULATION (Right Column)                   -->
  <!-- ========================================================================= -->
  <g transform="translate(1830, 130)">
    <rect x="0" y="0" width="510" height="1220" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2" filter="url(#card-shadow)" />
    <rect x="0" y="0" width="510" height="42" rx="12" fill="#f1f5f9" />
    <path d="M 0 30 L 0 42 L 510 42 L 510 30 Z" fill="#f1f5f9" />
    <text x="18" y="26" font-size="15" font-weight="700" fill="#1e293b">4. TASK HEADS &amp; LOSS FORMULATION</text>
    <rect x="385" y="10" width="105" height="22" rx="4" fill="#ea580c" />
    <text x="393" y="25" font-size="11" font-weight="700" fill="#ffffff">Multi-Task Fine-tune</text>

    <!-- HEAD 1: DENSE 3D SEGMENTATION -->
    <g transform="translate(18, 56)">
      <rect x="0" y="0" width="474" height="230" rx="8" fill="#ffffff" stroke="#38bdf8" stroke-width="1.5" />
      <rect x="0" y="0" width="474" height="28" rx="8" fill="#e0f2fe" />
      <text x="14" y="19" font-size="12" font-weight="700" fill="#0369a1">HEAD 1: 3D Dense Tumor Segmentation (UnetOutBlock3D)</text>
      
      <text x="14" y="48" font-size="11" fill="#334155">Input: Final Decoder Features <tspan font-family="'Consolas', monospace">[B, 48, 96, 96, 96]</tspan></text>
      <text x="14" y="66" font-size="11" fill="#334155">Architecture: <tspan font-weight="700">Conv3D (48 → 3, kernel=1³, stride=1, bias=True)</tspan></text>
      
      <!-- Channels Output -->
      <g transform="translate(14, 76)">
        <rect x="0" y="0" width="446" height="66" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1" />
        <text x="10" y="18" font-size="11" font-weight="700" fill="#0284c7">Output Channels (Multi-Label Sigmoid Activation):</text>
        <text x="10" y="34" font-size="10" fill="#0369a1">• Channel 0: Tumor Core (TC) = Necrotic Core + Enhancing Tumor</text>
        <text x="10" y="47" font-size="10" fill="#0369a1">• Channel 1: Whole Tumor (WT) = Complete Edema + Core + Enhancing</text>
        <text x="10" y="60" font-size="10" fill="#0369a1">• Channel 2: Enhancing Tumor (ET) = Active Contrast Enhancement</text>
      </g>

      <rect x="14" y="150" width="446" height="34" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="22" y="172" font-size="11" font-family="'Consolas', monospace" font-weight="700" fill="#0369a1">Raw Segmentation Logits: [B, 3, 96, 96, 96]</text>

      <!-- Loss formulation -->
      <g transform="translate(14, 190)">
        <text x="0" y="14" font-size="10" font-weight="700" fill="#c2410c">Loss: L_seg = L_Dice (weights: TC=1.3, WT=1.0, ET=1.5) + L_Focal (α=0.25, γ=2.0)</text>
      </g>
    </g>

    <!-- HEAD 2: GLIOMA GRADE CLASSIFICATION -->
    <g transform="translate(18, 305)">
      <rect x="0" y="0" width="474" height="245" rx="8" fill="#ffffff" stroke="#8b5cf6" stroke-width="1.5" />
      <rect x="0" y="0" width="474" height="28" rx="8" fill="#ede9fe" />
      <text x="14" y="19" font-size="12" font-weight="700" fill="#5b21b6">HEAD 2: Glioma Grade Classification (ClassificationHead)</text>

      <text x="14" y="48" font-size="11" fill="#334155">Input: Bottleneck Latent Representation <tspan font-family="'Consolas', monospace">[B, 768, 6, 6, 6]</tspan></text>
      
      <!-- Dual Pooling Structure -->
      <g transform="translate(14, 58)">
        <rect x="0" y="0" width="218" height="42" rx="4" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1" />
        <text x="10" y="18" font-size="10" font-weight="700" fill="#6b21a8">Global Avg Pooling (GAP)</text>
        <text x="10" y="32" font-size="9" font-family="'Consolas', monospace" fill="#4c1d95">AdaptiveAvgPool3D → [B, 768]</text>

        <rect x="228" y="0" width="218" height="42" rx="4" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1" />
        <text x="10" y="18" font-size="10" font-weight="700" fill="#6b21a8">Global Max Pooling (GMP)</text>
        <text x="10" y="32" font-size="9" font-family="'Consolas', monospace" fill="#4c1d95">AdaptiveMaxPool3D → [B, 768]</text>
      </g>

      <g transform="translate(14, 108)">
        <rect x="0" y="0" width="446" height="52" rx="4" fill="#f5f3ff" stroke="#c4b5fd" stroke-width="1" />
        <text x="10" y="16" font-size="11" font-weight="700" fill="#4c1d95">Fused Bottleneck Representation: [B, 1536]</text>
        <text x="10" y="32" font-size="10" fill="#334155">Linear(1536→256) → LN → GELU → <tspan font-weight="700" fill="#dc2626">Dropout(0.4)</tspan></text>
        <text x="10" y="45" font-size="10" fill="#334155">Linear(256→128) → LN → GELU → <tspan font-weight="700" fill="#dc2626">Dropout(0.2)</tspan> → Linear(128→2)</text>
      </g>

      <rect x="14" y="168" width="446" height="34" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="20" y="190" font-size="11" font-family="'Consolas', monospace" font-weight="700" fill="#5b21b6">Classification Logits: [B, 2] → Softmax [P(LGG), P(HGG)]</text>

      <g transform="translate(14, 210)">
        <text x="0" y="14" font-size="10" font-weight="700" fill="#c2410c">Loss: L_clf = CrossEntropy(label_smoothing=0.05)</text>
        <text x="0" y="28" font-size="10" font-weight="700" fill="#991b1b">Total Multi-Task Loss: L_total = L_seg + 0.5 · L_clf</text>
      </g>
    </g>

    <!-- SECTION 5: POST-PROCESSING & INFERENCE EVALUATION -->
    <g transform="translate(18, 570)">
      <rect x="0" y="0" width="474" height="630" rx="8" fill="#ffffff" stroke="#10b981" stroke-width="1.5" />
      <rect x="0" y="0" width="474" height="28" rx="8" fill="#ecfdf5" />
      <text x="14" y="19" font-size="12" font-weight="700" fill="#065f46">5. CLINICAL POST-PROCESSING &amp; TEST EVALUATION</text>

      <!-- Post-processing steps -->
      <g transform="translate(14, 40)">
        <text x="0" y="16" font-size="12" font-weight="700" fill="#0f172a">Post-Processing (src/utils/postprocessing.py):</text>
        
        <rect x="0" y="26" width="446" height="42" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="10" y="44" font-size="11" font-weight="700" fill="#15803d">1. Sigmoid Thresholding:</text>
        <text x="10" y="58" font-size="10" fill="#166534">Binary voxel segmentation map: p(voxel) &gt; 0.5 → {0, 1}</text>

        <rect x="0" y="74" width="446" height="74" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
        <text x="10" y="92" font-size="11" font-weight="700" fill="#15803d">2. 3D Connected Component Analysis (scipy.ndimage.label):</text>
        <text x="10" y="106" font-size="10" fill="#166534">Prunes false-positive isolated islands via anatomical thresholds:</text>
        <text x="10" y="122" font-size="10" font-weight="700" fill="#047857">• Whole Tumor (WT) cluster &lt; 250 voxels → 0</text>
        <text x="10" y="136" font-size="10" font-weight="700" fill="#047857">• Tumor Core (TC) &lt; 150 voxels → 0 | Enhancing (ET) &lt; 100 voxels → 0</text>
      </g>

      <!-- Evaluation Outputs -->
      <g transform="translate(14, 205)">
        <text x="0" y="16" font-size="12" font-weight="700" fill="#0f172a">Quantitative Evaluation Metrics (evaluate.py):</text>

        <rect x="0" y="26" width="446" height="120" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />
        <text x="10" y="44" font-size="11" font-weight="700" fill="#1e293b">Segmentation Metrics (Test Cohort: 251 Patients):</text>
        <text x="10" y="60" font-size="10" fill="#334155">• Overlap Accuracy: <tspan font-weight="700">Dice Score</tspan> (WT, TC, ET, and Mean Dice)</text>
        <text x="10" y="76" font-size="10" fill="#334155">• Boundary Distance: <tspan font-weight="700">95th Percentile Hausdorff Distance (HD95, mm)</tspan></text>
        <text x="10" y="92" font-size="10" fill="#334155">• Volumetric Conformance: Relative Volume Difference (RVD)</text>
        <text x="10" y="108" font-size="10" fill="#334155">• Sensitivity (Recall) &amp; Specificity across all 3 tumor subregions</text>

        <rect x="0" y="154" width="446" height="78" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />
        <text x="10" y="172" font-size="11" font-weight="700" fill="#1e293b">Classification Metrics (HGG vs LGG):</text>
        <text x="10" y="188" font-size="10" fill="#334155">• Accuracy (%) | F1-Score | Precision | Recall</text>
        <text x="10" y="204" font-size="10" fill="#334155">• Confusion Matrix (True Positive HGG, True Negative LGG)</text>
        <text x="10" y="220" font-size="10" fill="#334155">• Area Under the ROC Curve (AUC-ROC)</text>
      </g>

      <!-- Artifact Outputs -->
      <g transform="translate(14, 460)">
        <rect x="0" y="0" width="446" height="150" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5" />
        <text x="14" y="22" font-size="12" font-weight="700" fill="#14532d">Final Exportable Deliverables:</text>
        <text x="14" y="42" font-size="10" fill="#15803d">📁 <tspan font-weight="700">outputs/predictions/*.nii.gz</tspan> — Reconstructed 3D binary masks</text>
        <text x="14" y="60" font-size="10" fill="#15803d">📊 <tspan font-weight="700">outputs/graphs/01_to_10_*.png</tspan> — 10 Publication-grade evaluation plots</text>
        <text x="14" y="78" font-size="10" fill="#15803d">💾 <tspan font-weight="700">checkpoints/finetune/finetune_best.pt</tspan> — Optimal model checkpoint</text>
        <text x="14" y="96" font-size="10" fill="#15803d">🖥️ <tspan font-weight="700">app.py</tspan> — Interactive Clinical MRI Web Dashboard (Streamlit)</text>
        <text x="14" y="114" font-size="10" fill="#15803d">📓 <tspan font-weight="700">module_wise_output_demo.ipynb</tspan> — Step-by-step tensor verification</text>
        <text x="14" y="132" font-size="10" fill="#15803d">📄 <tspan font-weight="700">checkpoints/split/split_manifest.json</tspan> — Full audit log &amp; SHA-256 sums</text>
      </g>
    </g>
  </g>

  <!-- External Flow Connectors from Dec_out and Bottleneck to Task Heads -->
  <!-- Decoder Refined Output to Seg Head -->
  <path d="M 1740 500 L 1830 200" stroke="#0284c7" stroke-width="2.5" fill="none" marker-end="url(#arrow-blue)" />
  <!-- Bottleneck to Clf Head -->
  <path d="M 1485 860 L 1830 430" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="5,4" fill="none" marker-end="url(#arrow-purple)" />

</svg>
'''
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[OK] Generated SVG diagram at: {SVG_PATH}")


def create_drawio():
    """Generates standard Draw.io / diagrams.net XML file."""
    root = ET.Element("mxfile", host="app.diagrams.net", agent="Glioma-Architecture", version="21.0.0")
    diagram = ET.SubElement(root, "diagram", id="glioma_arch", name="Glioma Architecture")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="2000", dy="1400", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="2400", pageHeight="1400", background="#ffffff")
    root_cell = ET.SubElement(mxGraphModel, "root")
    
    # Layer 0 & 1
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    def add_cell(cid, value, style, x, y, w, h, parent="1"):
        cell = ET.SubElement(root_cell, "mxCell", id=str(cid), value=value, style=style, vertex="1", parent=parent)
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cell

    def add_edge(eid, source, target, style, value=""):
        edge = ET.SubElement(root_cell, "mxCell", id=str(eid), value=value, style=style, edge="1", source=str(source), target=str(target), parent="1")
        ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        return edge

    # Global title
    add_cell(10, "End-to-End System Architecture: 3D SwinMamba Dual &amp; IS-MIM Self-Supervised Learning",
             "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=24;fontStyle=1;fontColor=#0f172a;",
             60, 30, 1600, 40)
    add_cell(11, "Unified Deep Learning Pipeline for Multi-Parametric MRI Ingestion, 3D Intra-Modality Masked Pre-training, Joint Tumor Segmentation &amp; Glioma Grade Classification",
             "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#64748b;",
             60, 68, 1600, 25)

    # 1. Section 1 Container: Data Ingestion
    add_cell(100, "&lt;b&gt;1. DATA INGESTION &amp; SPLIT&lt;/b&gt;&lt;br&gt;BraTS 2023 Adult Glioma (1,251 Subjects)",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#cbd5e1;fontColor=#1e293b;strokeWidth=2;rounded=1;arcSize=4;fontSize=13;",
             60, 110, 320, 1250)
    
    add_cell(101, "&lt;b&gt;Raw Multi-Parametric MRI (3D NIfTI)&lt;/b&gt;&lt;br&gt;• T1-Native (t1n.nii)&lt;br&gt;• T1-Contrast (t1c.nii)&lt;br&gt;• T2-Weighted (t2w.nii)&lt;br&gt;• T2-FLAIR (t2f.nii)&lt;br&gt;• Ground Truth (seg.nii)",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f0f9ff;strokeColor=#0284c7;strokeWidth=1.5;fontColor=#0369a1;fontSize=11;align=left;spacingLeft=8;",
             80, 160, 280, 120)

    add_cell(102, "&lt;b&gt;Deterministic SplitManager&lt;/b&gt;&lt;br&gt;• Train: 750 (60.0%) Subjects&lt;br&gt;• Val: 250 (20.0%) Subjects&lt;br&gt;• Test: 251 (20.1%) Subjects (Locked)&lt;br&gt;Seed=42 | SHA-256 Checksums",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#64748b;strokeWidth=1.5;fontColor=#1e293b;fontSize=11;align=left;spacingLeft=8;",
             80, 310, 280, 110)

    add_cell(103, "&lt;b&gt;Preprocessing &amp; Normalization&lt;/b&gt;&lt;br&gt;1. Non-Zero Percentile Clip [0.5, 99.5]&lt;br&gt;2. Foreground Z-Score: (x - μ) / σ&lt;br&gt;3. Spatial Center Crop/Pad → [4, 96, 96, 96]&lt;br&gt;4. Label Map: WT(1+2+3), TC(1+3), ET(3)&lt;br&gt;5. Grade Inference: HGG(1) vs LGG(0)",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f0fdfa;strokeColor=#0d9488;strokeWidth=1.5;fontColor=#0f766e;fontSize=11;align=left;spacingLeft=8;",
             80, 450, 280, 140)

    add_cell(104, "&lt;b&gt;Batch Ingestion Tensors&lt;/b&gt;&lt;br&gt;&lt;font face='Consolas'&gt;Image: [B, 4, 96, 96, 96]&lt;br&gt;Seg GT: [B, 3, 96, 96, 96]&lt;br&gt;Label GT: [B] (0 or 1)&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#475569;strokeWidth=2;fontColor=#0f172a;fontSize=11;align=center;",
             80, 620, 280, 90)

    add_edge(1001, 101, 102, "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#475569;strokeWidth=2;")
    add_edge(1002, 102, 103, "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#475569;strokeWidth=2;")
    add_edge(1003, 103, 104, "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#475569;strokeWidth=2;")

    # 2. Section 2 Container: IS-MIM Pretraining
    add_cell(200, "&lt;b&gt;2. SELF-SUPERVISED PRE-TRAINING: INTRA-MODALITY MASKED IMAGE MODELING (IS-MIM)&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#c084fc;fontColor=#6b21a8;strokeWidth=2;rounded=1;arcSize=4;fontSize=13;",
             410, 110, 1380, 190)

    add_cell(201, "&lt;b&gt;2a. 3D Token Partition&lt;/b&gt;&lt;br&gt;Patch Size: 16³&lt;br&gt;96³ → 6×6×6 grid&lt;br&gt;216 tokens [B, 4, 16³, 216]",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#a855f7;strokeWidth=1.5;fontColor=#581c87;fontSize=11;",
             430, 160, 220, 110)

    add_cell(202, "&lt;b&gt;2b. Intra-Modality Swap&lt;/b&gt;&lt;br&gt;p_swap = 0.5 (2 iterations)&lt;br&gt;T1 ↔ T1ce, T2 ↔ FLAIR&lt;br&gt;Learns invariant anatomy",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#a855f7;strokeWidth=1.5;fontColor=#581c87;fontSize=11;",
             680, 160, 230, 110)

    add_cell(203, "&lt;b&gt;2c. 50% Block Masking&lt;/b&gt;&lt;br&gt;Mask Ratio = 50%&lt;br&gt;108 / 216 tokens zeroed&lt;br&gt;Generates voxel_mask M",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#a855f7;strokeWidth=1.5;fontColor=#581c87;fontSize=11;",
             940, 160, 220, 110)

    add_cell(204, "&lt;b&gt;2d. Detokenization&lt;/b&gt;&lt;br&gt;Reconstruct to 3D volume&lt;br&gt;X_perturbed: [B, 4, 96³]&lt;br&gt;X_clean: [B, 4, 96³]",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#a855f7;strokeWidth=1.5;fontColor=#581c87;fontSize=11;",
             1190, 160, 220, 110)

    add_cell(205, "&lt;b&gt;2e. Pre-train Objective&lt;/b&gt;&lt;br&gt;Masked MSE Loss&lt;br&gt;L_MIM = Σ M⊙(X̂-X)² / Σ M&lt;br&gt;AdamW, lr=5e-4",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff7ed;strokeColor=#ea580c;strokeWidth=1.5;fontColor=#9a3412;fontSize=11;",
             1440, 160, 230, 110)

    add_edge(2001, 201, 202, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#9333ea;strokeWidth=2;")
    add_edge(2002, 202, 203, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#9333ea;strokeWidth=2;")
    add_edge(2003, 203, 204, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#9333ea;strokeWidth=2;")
    add_edge(2004, 204, 205, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#9333ea;strokeWidth=2;")

    # 3. Section 3 Container: SwinMamba Dual Backbone & UNETR Decoder
    add_cell(300, "&lt;b&gt;3. CORE DEEP LEARNING MODEL: 3D HYBRID SWINMAMBA DUAL &amp; UNETR DECODER&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cbd5e1;fontColor=#0f172a;strokeWidth=2;rounded=1;arcSize=4;fontSize=14;",
             410, 330, 1380, 1030)

    # 3a. Hybrid Encoder Column
    add_cell(310, "&lt;b&gt;3A. HYBRID 3D ENCODER (SwinTransformer + MambaSSM)&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#818cf8;fontColor=#3730a3;strokeWidth=1.5;rounded=1;fontSize=12;",
             430, 380, 500, 950)

    add_cell(311, "&lt;b&gt;Stem Downsampler&lt;/b&gt;&lt;br&gt;Conv3D (4→48, k=7³, s=2, p=3) + BN + LeakyReLU + Dropout3D&lt;br&gt;&lt;font face='Consolas'&gt;Output: [B, 48, 48, 48, 48]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#6366f1;strokeWidth=1;fontColor=#1e1b4b;fontSize=10;",
             450, 420, 460, 50)

    add_cell(312, "&lt;b&gt;Stage 0: 3D Swin Transformer (Depth=2, Dims=48, Heads=3, Win=7³)&lt;/b&gt;&lt;br&gt;• Block 0: Window Attention (W-MSA)&lt;br&gt;• Block 1: Shifted Window Attention (SW-MSA, shift=(3,3,3))&lt;br&gt;Norm + MlpChannel (48→96→48) → &lt;b&gt;Skip S0 [B, 48, 48³]&lt;/b&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f3ff;strokeColor=#4f46e5;strokeWidth=1.5;fontColor=#312e81;fontSize=10;align=left;spacingLeft=8;",
             450, 490, 460, 80)

    add_cell(313, "&lt;b&gt;Patch Merging 0→1&lt;/b&gt;: Conv3D (48→96, k=2³, s=2) → [B, 96, 24³]",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#94a3b8;fontColor=#334155;fontSize=10;",
             450, 590, 460, 30)

    add_cell(314, "&lt;b&gt;Stage 1: 3D Swin Transformer (Depth=2, Dims=96, Heads=6, Win=7³)&lt;/b&gt;&lt;br&gt;• Block 0: Window Attention (W-MSA)&lt;br&gt;• Block 1: Shifted Window Attention (SW-MSA, shift=(3,3,3))&lt;br&gt;Norm + MlpChannel (96→192→96) → &lt;b&gt;Skip S1 [B, 96, 24³]&lt;/b&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f3ff;strokeColor=#4f46e5;strokeWidth=1.5;fontColor=#312e81;fontSize=10;align=left;spacingLeft=8;",
             450, 640, 460, 80)

    add_cell(315, "&lt;b&gt;Patch Merging 1→2&lt;/b&gt;: Conv3D (96→192, k=2³, s=2) → [B, 192, 12³]",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#94a3b8;fontColor=#334155;fontSize=10;",
             450, 740, 460, 30)

    add_cell(316, "&lt;b&gt;Stage 2: GSC + Tri-Directional Mamba SSM (Depth=2, Dims=192)&lt;/b&gt;&lt;br&gt;• GSC: Global Spatial Context Residual Convolutions (3×3×3 &amp; 1×1×1)&lt;br&gt;• Tri-Directional Selective Scan: Axial (D→H→W) + Sagittal (W→D→H) + Coronal (H→W→D)&lt;br&gt;Norm + MlpChannel (192→384→192) → &lt;b&gt;Skip S2 [B, 192, 12³]&lt;/b&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#7c3aed;strokeWidth=1.5;fontColor=#4c1d95;fontSize=10;align=left;spacingLeft=8;",
             450, 790, 460, 95)

    add_cell(317, "&lt;b&gt;Patch Merging 2→3&lt;/b&gt;: Conv3D (192→384, k=2³, s=2) → [B, 384, 6³]",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f5f9;strokeColor=#94a3b8;fontColor=#334155;fontSize=10;",
             450, 905, 460, 30)

    add_cell(318, "&lt;b&gt;Stage 3: Deepest Tri-Directional Mamba (Depth=2, Dims=384)&lt;/b&gt;&lt;br&gt;• GSC Spatial Prior + 3D SSM Continuous State Modeling&lt;br&gt;• Orthogonal Volumetric Sequences L = 6³ = 216 tokens&lt;br&gt;Output: &lt;b&gt;Deep Latents [B, 384, 6, 6, 6]&lt;/b&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#7c3aed;strokeWidth=1.5;fontColor=#4c1d95;fontSize=10;align=left;spacingLeft=8;",
             450, 955, 460, 80)

    add_edge(3001, 311, 312, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3002, 312, 313, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3003, 313, 314, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3004, 314, 315, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3005, 315, 316, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3006, 316, 317, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")
    add_edge(3007, 317, 318, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#475569;strokeWidth=1.5;")

    # 3b. Skip Encoders Column
    add_cell(330, "&lt;b&gt;3B. SKIP ENCODERS&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#f0f9ff;strokeColor=#38bdf8;fontColor=#0369a1;strokeWidth=1.5;rounded=1;fontSize=12;",
             960, 380, 200, 950)

    add_cell(331, "&lt;b&gt;enc_stem&lt;/b&gt;&lt;br&gt;4 → 48 dims&lt;br&gt;&lt;font face='Consolas'&gt;[B, 48, 96³]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0284c7;fontSize=10;",
             975, 420, 170, 50)
    add_cell(332, "&lt;b&gt;enc_stage0&lt;/b&gt;&lt;br&gt;48 → 48 dims&lt;br&gt;&lt;font face='Consolas'&gt;[B, 48, 48³]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0284c7;fontSize=10;",
             975, 510, 170, 50)
    add_cell(333, "&lt;b&gt;enc_stage1&lt;/b&gt;&lt;br&gt;96 → 96 dims&lt;br&gt;&lt;font face='Consolas'&gt;[B, 96, 24³]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0284c7;fontSize=10;",
             975, 660, 170, 50)
    add_cell(334, "&lt;b&gt;enc_stage2&lt;/b&gt;&lt;br&gt;192 → 192 dims&lt;br&gt;&lt;font face='Consolas'&gt;[B, 192, 12³]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0284c7;fontSize=10;",
             975, 810, 170, 50)
    add_cell(335, "&lt;b&gt;enc_bottleneck&lt;/b&gt;&lt;br&gt;384 → 768 dims&lt;br&gt;&lt;font face='Consolas'&gt;&lt;b&gt;[B, 768, 6³]&lt;/b&gt;&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#eff6ff;strokeColor=#1d4ed8;strokeWidth=2;fontSize=11;fontColor=#1e40af;",
             975, 965, 170, 60)

    # 3c. Progressive UNETR Decoder Column
    add_cell(350, "&lt;b&gt;3C. UNETR DECODER + EV2 REFINEMENT&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#60a5fa;fontColor=#1e40af;strokeWidth=1.5;rounded=1;fontSize=12;",
             1190, 380, 580, 950)

    add_cell(354, "&lt;b&gt;Decoder 4 (UnetrUpBlock3D)&lt;/b&gt;&lt;br&gt;ConvTranspose3D (768→192, k=2, s=2) ⊕ &lt;b&gt;skip_2 (192)&lt;/b&gt;&lt;br&gt;EV2Block Fusion (384→192) → &lt;font face='Consolas'&gt;[B, 192, 12, 12, 12]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2563eb;strokeWidth=1;fontSize=10;align=left;spacingLeft=8;",
             1210, 960, 540, 65)

    add_cell(353, "&lt;b&gt;Decoder 3 (UnetrUpBlock3D)&lt;/b&gt;&lt;br&gt;ConvTranspose3D (192→96, k=2, s=2) ⊕ &lt;b&gt;skip_1 (96)&lt;/b&gt;&lt;br&gt;EV2Block Fusion (192→96) → &lt;font face='Consolas'&gt;[B, 96, 24, 24, 24]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2563eb;strokeWidth=1;fontSize=10;align=left;spacingLeft=8;",
             1210, 810, 540, 65)

    add_cell(352, "&lt;b&gt;Decoder 2 (UnetrUpBlock3D)&lt;/b&gt;&lt;br&gt;ConvTranspose3D (96→48, k=2, s=2) ⊕ &lt;b&gt;skip_0 (48)&lt;/b&gt;&lt;br&gt;EV2Block Fusion (96→48) → &lt;font face='Consolas'&gt;[B, 48, 48, 48, 48]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2563eb;strokeWidth=1;fontSize=10;align=left;spacingLeft=8;",
             1210, 660, 540, 65)

    add_cell(351, "&lt;b&gt;Decoder 1 (UnetrUpBlock3D)&lt;/b&gt;&lt;br&gt;ConvTranspose3D (48→48, k=2, s=2) ⊕ &lt;b&gt;skip_stem (48)&lt;/b&gt;&lt;br&gt;EV2Block Fusion (96→48) → &lt;font face='Consolas'&gt;[B, 48, 96, 96, 96]&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2563eb;strokeWidth=1;fontSize=10;align=left;spacingLeft=8;",
             1210, 510, 540, 65)

    add_cell(3501, "&lt;b&gt;EV2 Residual Refinement Block (Final Latent)&lt;/b&gt;&lt;br&gt;Squeeze-and-Excitation (SE) Channel Attention + Residual Conv&lt;br&gt;&lt;b&gt;Output: dec_out [B, 48, 96, 96, 96]&lt;/b&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ecfdf5;strokeColor=#059669;strokeWidth=2;fontColor=#047857;fontSize=11;align=left;spacingLeft=8;",
             1210, 420, 540, 60)

    add_edge(3054, 354, 353, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#2563eb;strokeWidth=2;")
    add_edge(3053, 353, 352, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#2563eb;strokeWidth=2;")
    add_edge(3052, 352, 351, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#2563eb;strokeWidth=2;")
    add_edge(3051, 351, 3501, "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#059669;strokeWidth=2;")

    # 4. Section 4: Task Heads & Inference Evaluation
    add_cell(400, "&lt;b&gt;4. TASK HEADS &amp; INFERENCE EVALUATION&lt;/b&gt;",
             "swimlane;whiteSpace=wrap;html=1;fillColor=#f8fafc;strokeColor=#cbd5e1;fontColor=#1e293b;strokeWidth=2;rounded=1;arcSize=4;fontSize=13;",
             1830, 110, 510, 1250)

    add_cell(410, "&lt;b&gt;HEAD 1: 3D Tumor Segmentation&lt;/b&gt;&lt;br&gt;UnetOutBlock3D: Conv3D (48→3, k=1³)&lt;br&gt;Logits: [B, 3, 96³] → Sigmoid Probs&lt;br&gt;&lt;b&gt;Channels:&lt;/b&gt; TC (Core), WT (Whole), ET (Enhancing)&lt;br&gt;&lt;font color='#c2410c'&gt;&lt;b&gt;Loss:&lt;/b&gt; SoftDice (1.3, 1.0, 1.5) + Focal (α=0.25, γ=2)&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f0f9ff;strokeColor=#0284c7;strokeWidth=1.5;fontColor=#0369a1;fontSize=11;align=left;spacingLeft=8;",
             1850, 160, 470, 110)

    add_cell(420, "&lt;b&gt;HEAD 2: Glioma Grade Classification&lt;/b&gt;&lt;br&gt;Dual Global Pooling: GAP [B,768] + GMP [B,768]&lt;br&gt;Fused Vector: [B, 1536]&lt;br&gt;MLP: 1536 → 256 (Drop 0.4) → 128 (Drop 0.2) → 2&lt;br&gt;Logits: [B, 2] → Softmax [P(LGG), P(HGG)]&lt;br&gt;&lt;font color='#c2410c'&gt;&lt;b&gt;Loss:&lt;/b&gt; CrossEntropy(smoothing=0.05)&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#7c3aed;strokeWidth=1.5;fontColor=#4c1d95;fontSize=11;align=left;spacingLeft=8;",
             1850, 290, 470, 120)

    add_cell(430, "&lt;b&gt;Total Multi-Task Fine-Tuning Loss:&lt;/b&gt;&lt;br&gt;&lt;font face='Consolas' size='3'&gt;&lt;b&gt;L_total = L_Dice + L_Focal + 0.5 · L_CrossEntropy&lt;/b&gt;&lt;/font&gt;",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff7ed;strokeColor=#ea580c;strokeWidth=2;fontColor=#9a3412;fontSize=11;align=center;",
             1850, 430, 470, 50)

    add_cell(440, "&lt;b&gt;5. Clinical Post-Processing Pipeline&lt;/b&gt;&lt;br&gt;1. Sigmoid Thresholding (p &gt; 0.5)&lt;br&gt;2. 3D Connected Component Analysis (scipy.ndimage):&lt;br&gt;   • WT clusters &lt; 250 voxels removed&lt;br&gt;   • TC clusters &lt; 150 voxels removed&lt;br&gt;   • ET clusters &lt; 100 voxels removed",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#f0fdf4;strokeColor=#059669;strokeWidth=1.5;fontColor=#047857;fontSize=11;align=left;spacingLeft=8;",
             1850, 510, 470, 110)

    add_cell(450, "&lt;b&gt;Quantitative Evaluation on Held-Out Test Set (251 Cases)&lt;/b&gt;&lt;br&gt;• Segmentation: Dice Scores (WT, TC, ET, Mean), HD95 (mm)&lt;br&gt;• Classification: Accuracy, F1-Score, Precision, Recall, AUC-ROC&lt;br&gt;• Checkpoint Verification: finetune_best.pt&lt;br&gt;• NIfTI Binary Predictions: outputs/predictions/*.nii.gz",
             "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#10b981;strokeWidth=1.5;fontColor=#065f46;fontSize=11;align=left;spacingLeft=8;",
             1850, 640, 470, 100)

    tree = ET.ElementTree(root)
    tree.write(DRAWIO_PATH, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Generated Draw.io file at: {DRAWIO_PATH}")

if __name__ == "__main__":
    create_svg()
    create_drawio()
