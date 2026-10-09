#!/usr/bin/env python3
"""
Generate Publication-Grade Architecture Diagram for Layer Normalization Foundation & 2024-2026 Survey
===================================================================================================
Produces:
  1. docs/diagrams/layernorm_evolution_2026.excalidraw
  2. docs/diagrams/layernorm_evolution_2026.excalidraw.md (Obsidian-compatible)
  3. docs/diagrams/layernorm_evolution_2026.png
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
LATEX_CACHE_DIR = DIAGRAMS_DIR / ".latex_cache"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
LATEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def generate_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

def render_latex_to_svg(latex_str: str, file_prefix: str, fontsize: int = 16, color: str = "#38bdf8", bg_color: str = "#0f172a") -> Path:
    """Render LaTeX formula to SVG using Matplotlib."""
    clean_hash = hashlib.md5(f"{latex_str}_{fontsize}_{color}_{bg_color}".encode('utf-8')).hexdigest()[:8]
    svg_filename = f"{file_prefix}_{clean_hash}.svg"
    svg_path = LATEX_CACHE_DIR / svg_filename
    
    if svg_path.exists():
        return svg_path

    fig = plt.figure(figsize=(0.1, 0.1), dpi=300)
    fig.patch.set_facecolor(bg_color)
    fig.patch.set_alpha(0.0)
    
    text = plt.text(
        0.5, 0.5, f"${latex_str}$",
        fontsize=fontsize,
        color=color,
        ha='center', va='center',
        usetex=False
    )
    plt.axis('off')
    
    fig.canvas.draw()
    bbox = text.get_window_extent(fig.canvas.get_renderer())
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    
    pad = 0.08
    bbox_inches_padded = bbox_inches.expanded(1.0 + pad, 1.0 + pad)
    
    plt.savefig(
        str(svg_path),
        format='svg',
        bbox_inches=bbox_inches_padded,
        transparent=True,
        pad_inches=0.02
    )
    plt.close(fig)
    return svg_path

def render_vector_illustration(render_func, filename: str, figsize=(4.2, 1.4), dpi=220) -> Path:
    """Render a dedicated custom vector illustration to SVG."""
    svg_path = LATEX_CACHE_DIR / filename
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0f172a')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#0f172a')
    ax.set_alpha(0.0)
    ax.axis('off')
    
    render_func(ax)
    
    plt.tight_layout(pad=0.05)
    plt.savefig(str(svg_path), format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return svg_path

# ==========================================
# CUSTOM VECTOR MINI-ILLUSTRATIONS
# ==========================================

def render_mini_ln_geometry(ax):
    """Draw Sphere S^{d-1} and Orthogonal Projection J_LN."""
    import numpy as np
    theta = np.linspace(0, 2*np.pi, 200)
    # Circle representing unit sphere slice
    ax.plot(np.cos(theta), np.sin(theta), color='#38bdf8', lw=2, ls='--', alpha=0.8)
    
    # 1 vector line (mean direction)
    ax.plot([-1.4, 1.4], [-1.4, 1.4], color='#ef4444', lw=1.8, ls=':', label='1 vector (Mean direction)')
    
    # Unnormalized vector x
    ax.annotate('', xy=(0.8, 1.6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='#f59e0b', lw=2.5))
    ax.text(0.85, 1.65, r'$\mathbf{x} \in \mathbb{R}^d$', color='#fbbf24', fontsize=9, fontweight='bold')
    
    # Mean-centered vector x - mu 1 (orthogonal to 1)
    ax.annotate('', xy=(1.1, -0.1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='#a855f7', lw=2))
    ax.text(1.15, -0.25, r'$\mathbf{x} - \mu\mathbf{1} \perp \mathbf{1}$', color='#c084fc', fontsize=8.5)
    
    # Normalized vector on sphere
    ax.scatter([0.95], [-0.09], color='#10b981', s=50, zorder=5)
    ax.text(0.7, 0.15, r'$\hat{\mathbf{x}} \in \mathbb{S}^{d-1}$', color='#34d399', fontsize=9, fontweight='bold')
    
    # Projection arrow
    ax.annotate('', xy=(0.95, -0.09), xytext=(0.8, 1.6),
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=1.5, ls='--'))
    ax.text(0.1, 0.9, r'$\mathbf{P}^\perp_{\mathbf{1}} / \sigma$', color='#38bdf8', fontsize=8)
    
    ax.set_xlim(-1.6, 1.8)
    ax.set_ylim(-1.6, 1.8)

def render_mini_norm_taxonomy(ax):
    """Draw BatchNorm vs LayerNorm vs RMSNorm vs GroupNorm 3D tensor slices."""
    import matplotlib.patches as patches
    # Batch dimension (N=4), Channels (C=4), Spatial/Sequence (L=4)
    # Draw 4 small subplots conceptually
    # 1. BN: across N
    # 2. LN: across (C, L) or D
    # 3. RMSNorm: across D (no mean)
    # 4. GN: across group of C
    
    rect_style = dict(ec='#334155', lw=1, alpha=0.9)
    # BN Box
    ax.add_patch(patches.Rectangle((0.1, 0.2), 0.7, 1.0, fc='#1e293b', **rect_style))
    ax.add_patch(patches.Rectangle((0.2, 0.2), 0.15, 1.0, fc='#3b82f6', alpha=0.8)) # slice across batch
    ax.text(0.45, 1.3, 'BatchNorm', color='#60a5fa', fontsize=8, ha='center', fontweight='bold')
    ax.text(0.45, 0.05, r'$(B \times S) \to \text{Dep. } B$', color='#94a3b8', fontsize=6.5, ha='center')

    # LN Box
    ax.add_patch(patches.Rectangle((1.1, 0.2), 0.7, 1.0, fc='#1e293b', **rect_style))
    ax.add_patch(patches.Rectangle((1.1, 0.8), 0.7, 0.2, fc='#10b981', alpha=0.8)) # slice across hidden D
    ax.text(1.45, 1.3, 'LayerNorm', color='#34d399', fontsize=8, ha='center', fontweight='bold')
    ax.text(1.45, 0.05, r'$(D) \to \mu, \sigma^2$', color='#94a3b8', fontsize=6.5, ha='center')

    # RMSNorm Box
    ax.add_patch(patches.Rectangle((2.1, 0.2), 0.7, 1.0, fc='#1e293b', **rect_style))
    ax.add_patch(patches.Rectangle((2.1, 0.8), 0.7, 0.2, fc='#a855f7', alpha=0.8)) # slice across D (RMS)
    ax.text(2.45, 1.3, 'RMSNorm', color='#c084fc', fontsize=8, ha='center', fontweight='bold')
    ax.text(2.45, 0.05, r'$(D) \to \text{RMS}, \text{No } \mu$', color='#94a3b8', fontsize=6.5, ha='center')

    # Dynamic Tanh DyT
    ax.add_patch(patches.Rectangle((3.1, 0.2), 0.7, 1.0, fc='#1e293b', **rect_style))
    ax.plot([3.2, 3.45, 3.65, 3.75], [0.35, 0.5, 0.9, 1.05], color='#f59e0b', lw=2.5)
    ax.text(3.45, 1.3, 'Dynamic Tanh (DyT)', color='#fbbf24', fontsize=8, ha='center', fontweight='bold')
    ax.text(3.45, 0.05, r'$\tanh(\alpha x) \to \text{No Stats}$', color='#94a3b8', fontsize=6.5, ha='center')

    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.1, 1.5)

def render_mini_pre_post_ln(ax):
    """Draw Post-LN vs Pre-LN vs DeepNorm gradient propagation comparison."""
    import matplotlib.patches as patches
    # Post-LN
    ax.add_patch(patches.Rectangle((0.1, 0.5), 0.8, 0.4, fc='#ef4444', alpha=0.3, ec='#ef4444', lw=1.5))
    ax.text(0.5, 0.7, 'Post-LN\n(LN on Skip)', color='#fca5a5', fontsize=7.5, ha='center', va='center')
    ax.text(0.5, 0.15, r'$\nabla \mathcal{L} \to \text{Decays } \mathcal{O}(L^{-1})$', color='#f87171', fontsize=7, ha='center')
    
    # Pre-LN
    ax.add_patch(patches.Rectangle((1.2, 0.5), 0.8, 0.4, fc='#10b981', alpha=0.3, ec='#10b981', lw=1.5))
    ax.text(1.6, 0.7, 'Pre-LN\n(LN inside Branch)', color='#86efac', fontsize=7.5, ha='center', va='center')
    ax.text(1.6, 0.15, r'$\nabla \mathcal{L} \to \mathbf{I} + \dots \text{ (Highway)}$', color='#4ade80', fontsize=7, ha='center')

    # DeepNorm
    ax.add_patch(patches.Rectangle((2.3, 0.5), 0.8, 0.4, fc='#38bdf8', alpha=0.3, ec='#38bdf8', lw=1.5))
    ax.text(2.7, 0.7, 'DeepNorm\n(Alpha Scaled Skip)', color='#bae6fd', fontsize=7.5, ha='center', va='center')
    ax.text(2.7, 0.15, r'$\text{Stable up to } 1000 \text{ Layers}$', color='#38bdf8', fontsize=7, ha='center')

    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 1.1)

def render_mini_qknorm(ax):
    """Draw QK-Norm mechanism preventing Attention Logit Explosion."""
    import matplotlib.patches as patches
    import numpy as np
    
    # Softmax entropy curve
    x = np.linspace(-3, 3, 100)
    # Unbounded dot product
    ax.plot(x, 1/(1+np.exp(-3*x)), color='#ef4444', lw=2, ls='--', label='Unbounded QK (Spiky Softmax)')
    # QK-Norm bounded
    ax.plot(x, 1/(1+np.exp(-x)), color='#38bdf8', lw=2.5, label='QK-Norm (Smooth Entropy)')
    
    ax.text(0, 1.15, r'$\mathbf{A} = \text{softmax}\left(\frac{\text{Norm}(\mathbf{Q})\text{Norm}(\mathbf{K})^T}{\sqrt{d}}\right)$',
            color='#38bdf8', fontsize=8.5, ha='center')
    ax.text(1.8, 0.2, 'Without QK-Norm\nLogit Explosion', color='#ef4444', fontsize=7)
    ax.text(-2.5, 0.8, 'With QK-Norm\nStable FP8 / 1M ctx', color='#34d399', fontsize=7)
    
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.1, 1.35)

def render_mini_adaln(ax):
    """Draw AdaLN-Zero in Diffusion Transformers (DiT)."""
    import matplotlib.patches as patches
    
    # Conditioning Vector c
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.6, 0.5, fc='#6366f1', ec='#818cf8', lw=1.5))
    ax.text(0.4, 0.65, r'$\mathbf{c} \in \mathbb{R}^k$', color='#e0e7ff', fontsize=8.5, ha='center', va='center')
    ax.text(0.4, 0.2, 'Condition (t, y)', color='#a5b4fc', fontsize=7, ha='center')

    # MLP projection
    ax.annotate('', xy=(1.2, 0.65), xytext=(0.7, 0.65),
                arrowprops=dict(arrowstyle="->", color='#818cf8', lw=2))
    ax.text(0.95, 0.8, 'MLP', color='#c7d2fe', fontsize=7.5, ha='center')

    # Scale/Shift/Gate outputs
    ax.add_patch(patches.Rectangle((1.2, 0.2), 1.6, 0.9, fc='#1e293b', ec='#38bdf8', lw=1.5))
    ax.text(2.0, 0.85, r'$\boldsymbol{\gamma}_c, \boldsymbol{\beta}_c = \text{Scale / Shift}$', color='#38bdf8', fontsize=7.5, ha='center')
    ax.text(2.0, 0.55, r'$\boldsymbol{\alpha}_c = \text{Gate (Init } 0)$', color='#34d399', fontsize=7.5, ha='center')
    ax.text(2.0, 0.3, r'$\mathbf{y} = \boldsymbol{\alpha}_c \odot \mathcal{F}(\text{LN}(\mathbf{x}) \odot \boldsymbol{\gamma}_c + \boldsymbol{\beta}_c)$', color='#f59e0b', fontsize=7, ha='center')

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0.05, 1.25)

def render_mini_dyt_derf(ax):
    """Draw Point-wise S-curve for Dynamic Tanh (DyT) & Derf."""
    import numpy as np
    x = np.linspace(-3, 3, 100)
    y_tanh = np.tanh(1.2 * x)
    # erf
    from scipy.special import erf
    y_erf = erf(1.0 * x)
    
    ax.plot(x, y_tanh, color='#f59e0b', lw=2.2, label=r'$\text{DyT} = \tanh(\alpha x)$')
    ax.plot(x, y_erf, color='#ec4899', lw=2.2, ls='--', label=r'$\text{Derf} = \text{erf}(\alpha x + s)$')
    ax.axhline(0, color='#334155', lw=1, ls=':')
    ax.axvline(0, color='#334155', lw=1, ls=':')
    
    ax.text(0, 1.2, r'Zero Cross-Token Comm $\to \mathcal{O}(1)$ Bandwidth', color='#38bdf8', fontsize=8, ha='center', fontweight='bold')
    ax.text(1.8, 0.6, 'DyT (CVPR 2025)', color='#fbbf24', fontsize=7.5)
    ax.text(-2.8, -0.7, 'Derf (2026)', color='#f472b6', fontsize=7.5)
    
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.3, 1.45)

# ==========================================
# EXCALIDRAW ELEMENT BUILDERS
# ==========================================

def create_rectangle(id_seed, x, y, width, height, stroke_color="#38bdf8", bg_color="#0f172a", fill_style="solid", stroke_width=2, roughness=0, roundness={"type": 3}):
    return {
        "id": generate_id(id_seed),
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": bg_color,
        "fillStyle": fill_style,
        "strokeWidth": stroke_width,
        "strokeStyle": "solid",
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": roundness,
        "seed": abs(hash(id_seed)) % 1000000,
        "version": 1,
        "versionNonce": abs(hash(id_seed + "_vn")) % 1000000,
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False
    }

def create_text(id_seed, x, y, text, font_size=16, color="#f8fafc", font_family=1, text_align="left", width=None, height=None):
    lines = text.split("\n")
    calc_width = width if width is not None else max(len(l) for l in lines) * (font_size * 0.6)
    calc_height = height if height is not None else len(lines) * (font_size * 1.35)
    return {
        "id": generate_id(id_seed),
        "type": "text",
        "x": x,
        "y": y,
        "width": calc_width,
        "height": calc_height,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": abs(hash(id_seed)) % 1000000,
        "version": 1,
        "versionNonce": abs(hash(id_seed + "_vn")) % 1000000,
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
        "text": text,
        "fontSize": font_size,
        "fontFamily": font_family,
        "textAlign": text_align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": text,
        "lineHeight": 1.35
    }

def create_image_element(id_seed, x, y, width, height, file_id, latex_source=None):
    custom_data = {}
    if latex_source:
        custom_data["excalimath"] = {
            "latex": latex_source,
            "version": 1,
            "renderedAt": int(time.time() * 1000)
        }
    return {
        "id": generate_id(id_seed),
        "type": "image",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "transparent",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": abs(hash(id_seed)) % 1000000,
        "version": 1,
        "versionNonce": abs(hash(id_seed + "_vn")) % 1000000,
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
        "status": "saved",
        "fileId": file_id,
        "scale": [1, 1],
        "customData": custom_data if custom_data else None
    }

def create_arrow(id_seed, start_x, start_y, end_x, end_y, stroke_color="#38bdf8", stroke_width=2, stroke_style="solid"):
    dx = end_x - start_x
    dy = end_y - start_y
    return {
        "id": generate_id(id_seed),
        "type": "arrow",
        "x": start_x,
        "y": start_y,
        "width": abs(dx),
        "height": abs(dy),
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "strokeStyle": stroke_style,
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": abs(hash(id_seed)) % 1000000,
        "version": 1,
        "versionNonce": abs(hash(id_seed + "_vn")) % 1000000,
        "isDeleted": False,
        "boundElements": [],
        "updated": int(time.time() * 1000),
        "link": None,
        "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow"
    }

def build_layernorm_master_diagram():
    elements = []
    files_dict = {}

    # Helper to register an SVG file
    def register_svg(svg_path: Path, mime="image/svg+xml") -> str:
        with open(svg_path, "r", encoding="utf-8") as f:
            content = f.read()
        import base64
        b64_data = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        data_url = f"data:{mime};base64,{b64_data}"
        file_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]
        files_dict[file_id] = {
            "mimeType": mime,
            "id": file_id,
            "dataURL": data_url,
            "created": int(time.time() * 1000)
        }
        return file_id

    # Canvas Size: 2500 x 2600 px
    # Background Canvas
    elements.append(create_rectangle("canvas_bg", 0, 0, 2500, 2600, stroke_color="#1e293b", bg_color="#020617", stroke_width=1, roundness=None))

    # Top Badges
    badges = [
        ("badge_1", 40, 30, 220, 32, "⚡ FOUNDATION & GEOMETRY", "#38bdf8", "#0369a1"),
        ("badge_2", 280, 30, 200, 32, "📐 S^{d-1} SPHERE PROJECTION", "#a855f7", "#6b21a8"),
        ("badge_3", 500, 30, 220, 32, "🚀 2024-2026 SOTA SURVEY", "#10b981", "#047857"),
        ("badge_4", 740, 30, 240, 32, "🔬 HARDWARE & QUANT-AWARE", "#f59e0b", "#b45309"),
        ("badge_5", 1000, 30, 240, 32, "📚 OPENSCHOLAR RESEARCH", "#ec4899", "#be185d")
    ]
    for bid, bx, by, bw, bh, btext, bstroke, bbg in badges:
        elements.append(create_rectangle(bid, bx, by, bw, bh, stroke_color=bstroke, bg_color=bbg, stroke_width=1.5, roundness={"type": 3}))
        elements.append(create_text(bid + "_txt", bx + 12, by + 7, btext, font_size=12, color="#ffffff", font_family=1))

    # Master Title Header
    elements.append(create_text("title_main", 40, 75, "Layer Normalization: Foundations, Mathematical Rigor & 2024–2026 Modern SOTA Survey", font_size=24, color="#f8fafc", font_family=1))
    elements.append(create_text("title_sub", 40, 108, "From Hinton's 2016 Manifold Invariance to Pre/Post-LN, DeepNorm, RMSNorm, QK-Norm, AdaLN-Zero, and 2025/2026 Normalization-Free Frontiers (DyT, Derf)", font_size=13, color="#94a3b8", font_family=1))

    # =========================================================================
    # SECTION A: THE 4 CARDINAL NORMALIZATION PARADIGMS (y = 150 .. 680)
    # =========================================================================
    sec_a_box = create_rectangle("sec_a_frame", 40, 150, 2420, 520, stroke_color="#38bdf8", bg_color="#0b1329", stroke_width=2)
    elements.append(sec_a_box)
    elements.append(create_text("sec_a_title", 60, 168, "SECTION A: THE 4 CARDINAL NORMALIZATION PARADIGMS IN DEEP LEARNING", font_size=15, color="#38bdf8", font_family=1))

    # 4 Cards in Section A: BatchNorm, LayerNorm, RMSNorm, GroupNorm/InstanceNorm
    card_w = 580
    card_h = 430
    norm_cards = [
        ("A.1 BATCH NORMALIZATION (BN - 2015)", 60, 205, "#3b82f6",
         r"\mu_B = \frac{1}{m}\sum x_i, \quad \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}",
         "• Computes statistics across Batch dimension B.\n• Batch-size dependent (fails when B < 32 or dynamic seq).\n• Train/Inference mismatch (requires running mean/var).\n• Ruled out in modern Autoregressive Transformers / LLMs."),
        
        ("A.2 LAYER NORMALIZATION (LN - 2016)", 665, 205, "#10b981",
         r"\mu_L = \frac{1}{d}\sum_{i=1}^d x_i, \quad y = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu_L\mathbf{1}}{\sigma_L} + \boldsymbol{\beta}",
         "• Computes statistics across Feature dimension d independently.\n• Completely Batch-size agnostic (identical train/eval).\n• Invariant to scaling and shifting of inputs & weights.\n• Foundation for BERT, ViT, GPT-2/3, CLIP, DINOv2."),

        ("A.3 ROOT MEAN SQUARE NORM (RMSNorm - 2019)", 1270, 205, "#a855f7",
         r"\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \quad \mathbf{y} = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}",
         "• Removes Mean centering (μ = 0) and Bias term (β = 0).\n• 7% - 50% faster kernel speed due to reduced memory passes.\n• Assumes representations naturally orthogonal to 1.\n• De facto standard for LLaMA 3, Gemma 2, Mistral, Qwen 2.5."),

        ("A.4 GROUP / INSTANCE / DUAL NORM", 1875, 205, "#f59e0b",
         r"\text{GN}(\mathbf{x}) = \frac{\mathbf{x} - \mu_G}{\sigma_G}, \quad G \subset \{1 \dots d\} \text{ (Groups)}",
         "• Partitions channels into G groups (bridges BN and LN).\n• Preferred in ConvNeXt, Swin Transformer, and 2D/3D CNNs.\n• Dual-Norm: Channel-wise + Token-wise orthogonalization.\n• Crucial in high-resolution spatial feature stabilizers.")
    ]

    for ctitle, cx, cy, ccolor, clatex, cbullets in norm_cards:
        elements.append(create_rectangle(f"card_{cx}", cx, cy, card_w, card_h, stroke_color=ccolor, bg_color="#0f172a", stroke_width=1.5))
        elements.append(create_rectangle(f"card_head_{cx}", cx, cy, card_w, 40, stroke_color=ccolor, bg_color=ccolor, stroke_width=1, roundness={"type": 3}))
        elements.append(create_text(f"card_txt_{cx}", cx + 12, cy + 10, ctitle, font_size=12, color="#ffffff", font_family=1))
        
        # Render LaTeX formula
        l_svg = render_latex_to_svg(clatex, f"sec_a_card_{cx}", fontsize=14, color=ccolor)
        fid = register_svg(l_svg)
        elements.append(create_image_element(f"card_img_{cx}", cx + 15, cy + 50, card_w - 30, 48, fid, latex_source=clatex))
        
        # Bullets
        elements.append(create_text(f"card_bul_{cx}", cx + 18, cy + 110, cbullets, font_size=12, color="#e2e8f0", font_family=1, width=card_w - 36))

    # Add Section A Overview Illustration
    svg_a_overview = render_vector_illustration(render_mini_norm_taxonomy, "sec_a_taxonomy.svg", figsize=(10.5, 2.2), dpi=240)
    fid_a_ov = register_svg(svg_a_overview)
    elements.append(create_image_element("sec_a_ov_img", 120, 450, 2260, 200, fid_a_ov))

    # =========================================================================
    # SECTION B: DEEP MATHEMATICAL FOUNDATION & GEOMETRY (y = 690 .. 1320)
    # =========================================================================
    sec_b_box = create_rectangle("sec_b_frame", 40, 690, 2420, 610, stroke_color="#10b981", bg_color="#07171e", stroke_width=2)
    elements.append(sec_b_box)
    elements.append(create_text("sec_b_title", 60, 708, "SECTION B: MATHEMATICAL FOUNDATION, FORWARD/BACKWARD DYNAMICS & RIEMANNIAN GEOMETRY", font_size=15, color="#34d399", font_family=1))

    # 3 Cards in Section B
    col_w_b = 780
    b_cards = [
        ("B.1 INVARIANCE LAWS & AUTO-REGULARIZATION", 60, 745, "#10b981",
         r"\text{LN}(\alpha \mathbf{x} + c\mathbf{1}) = \text{LN}(\mathbf{x}), \quad \frac{\partial \text{LN}(\alpha\mathbf{x})}{\partial (\alpha\mathbf{x})} = \frac{1}{\alpha}\frac{\partial \text{LN}(\mathbf{x})}{\partial \mathbf{x}}",
         "• Shift Invariance: Cancels baseline DC drift across layers.\n• Scale Invariance: Output independent of input magnitude.\n• Weight Scale Invariance: If W -> a*W, LN(W x) = LN(a*W x).\n• Gradient Auto-Regulation: Larger weights -> smaller gradient step.\n• Acts as an intrinsic adaptive learning rate scheduler."),

        ("B.2 EXACT BACKWARD JACOBIAN & PROJECTION", 880, 745, "#38bdf8",
         r"\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} = \frac{1}{\sigma}\left(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^T\right) = \frac{1}{\sigma}\mathbf{P}^\perp_{\mathbf{1}}\mathbf{P}^\perp_{\hat{\mathbf{x}}}",
         "• Jacobian decomposes into double orthogonal projectors.\n• P_1 cancels mean gradient: sum_i (dL / dx_i) = 0.\n• P_x cancels radial gradient component along x.\n• Gradient flows strictly tangent to the hypersphere S^{d-1}.\n• Prevents internal covariate shift and stabilizes Hessian conditioning."),

        ("B.3 RIEMANNIAN MANIFOLD ON S^{d-1} & LOSS LANDSCAPE", 1700, 745, "#a855f7",
         r"\mathcal{M} = \mathbb{S}^{d-1} \subset \mathbb{R}^d, \quad \kappa(\mathbf{H}_{\text{LN}}) \ll \kappa(\mathbf{H}_{\text{Raw}})",
         "• Constrains hidden activations onto (d-1)-dim unit hypersphere.\n• Loss Surface Smoothing: Santurkar et al. & Xu et al. prove\n  LN makes loss Lipschitz continuous: ||grad L(x) - grad L(y)|| <= L ||x-y||.\n• Eliminates sharp ravines, enabling larger learning rates.\n• Ensures isotropic gradient flow without exploding curvature.")
    ]

    for ctitle, cx, cy, ccolor, clatex, cbullets in b_cards:
        elements.append(create_rectangle(f"card_b_{cx}", cx, cy, col_w_b, 530, stroke_color=ccolor, bg_color="#0f172a", stroke_width=1.5))
        elements.append(create_rectangle(f"card_b_head_{cx}", cx, cy, col_w_b, 40, stroke_color=ccolor, bg_color=ccolor, stroke_width=1, roundness={"type": 3}))
        elements.append(create_text(f"card_b_txt_{cx}", cx + 12, cy + 10, ctitle, font_size=12, color="#ffffff", font_family=1))
        
        # Render LaTeX formula
        l_svg = render_latex_to_svg(clatex, f"sec_b_card_{cx}", fontsize=13.5, color=ccolor)
        fid = register_svg(l_svg)
        elements.append(create_image_element(f"card_b_img_{cx}", cx + 15, cy + 50, col_w_b - 30, 48, fid, latex_source=clatex))
        
        # Bullets
        elements.append(create_text(f"card_b_bul_{cx}", cx + 18, cy + 110, cbullets, font_size=12, color="#e2e8f0", font_family=1, width=col_w_b - 36))

    # Add Mini Illustration inside B.2 and B.3
    svg_b_geo = render_vector_illustration(render_mini_ln_geometry, "sec_b_geometry.svg", figsize=(4.0, 1.8), dpi=220)
    fid_b_geo = register_svg(svg_b_geo)
    elements.append(create_image_element("sec_b_geo_img", 950, 970, 640, 280, fid_b_geo))

    svg_b_prepost = render_vector_illustration(render_mini_pre_post_ln, "sec_b_prepost.svg", figsize=(4.2, 1.8), dpi=220)
    fid_b_pp = register_svg(svg_b_prepost)
    elements.append(create_image_element("sec_b_pp_img", 1750, 970, 680, 280, fid_b_pp))

    # =========================================================================
    # SECTION C: TRANSFORMER PLACEMENT EVOLUTION & SCALING (y = 1320 .. 1930)
    # =========================================================================
    sec_c_box = create_rectangle("sec_c_frame", 40, 1320, 2420, 590, stroke_color="#a855f7", bg_color="#180b29", stroke_width=2)
    elements.append(sec_c_box)
    elements.append(create_text("sec_c_title", 60, 1338, "SECTION C: ARCHITECTURAL PLACEMENT EVOLUTION & DEEP STABILITY (PRE-LN, DEEPNORM, PARALLEL)", font_size=15, color="#c084fc", font_family=1))

    # 3 Cards in Section C
    c_cards = [
        ("C.1 POST-LN vs PRE-LN vs SANDWICH-LN", 60, 1375, "#a855f7",
         r"\text{Pre-LN: } \mathbf{x}_l = \mathbf{x}_{l-1} + \mathcal{F}(\text{LN}(\mathbf{x}_{l-1})), \quad \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \mathbf{I} + \sum \frac{\partial \mathcal{F}}{\partial \mathbf{x}}",
         "• Post-LN (Vaswani 2017): LN placed on skip output -> Expected gradient\n  decays as O(1/L), requiring critical warm-up phase.\n• Pre-LN (Xiong 2020): LN placed inside sub-layer branch ->\n  Creates an unobstructed Identity Gradient Highway (d x_L / d x_l = I + ...).\n• Sandwich-LN (CogView): Adds extra LN after sub-layer before skip,\n  preventing activation blowout in Vision-Language transformers."),

        ("C.2 DEEPNORM: 1,000-LAYER TRANSFORMERS", 880, 1375, "#38bdf8",
         r"\mathbf{x}_l = \text{LN}\left(\mathbf{x}_{l-1} \cdot \alpha + \mathcal{F}(\mathbf{x}_{l-1})\right), \quad \alpha = (2L)^{1/4}, \; \beta = (8L)^{-1/4}",
         "• Microsoft DeepNet (Wang et al., 2022) scales transformers to 1,000 layers.\n• Multiplies residual connection by alpha > 1 while scaling weights by beta < 1.\n• Bound activation variance: Var(x_L) = Theta(1) regardless of depth L.\n• Combines Post-LN expressivity with Pre-LN training stability."),

        ("C.3 PARALLEL TRANSFORMER BLOCKS", 1700, 1375, "#f59e0b",
         r"\mathbf{x}_l = \mathbf{x}_{l-1} + \text{Attn}(\text{LN}_1(\mathbf{x}_{l-1})) + \text{MLP}(\text{LN}_2(\mathbf{x}_{l-1}))",
         "• PaLM (Google 2022) & GPT-J: Runs Attention and MLP in parallel.\n• Shared LN: LN_1 = LN_2 -> fuses input projections into 1 single matrix mult.\n• 15% faster training throughput with zero quality degradation.\n• Supported by TensorRT-LLM and FlashAttention-3 unified kernel.")
    ]

    for ctitle, cx, cy, ccolor, clatex, cbullets in c_cards:
        elements.append(create_rectangle(f"card_c_{cx}", cx, cy, col_w_b, 510, stroke_color=ccolor, bg_color="#0f172a", stroke_width=1.5))
        elements.append(create_rectangle(f"card_c_head_{cx}", cx, cy, col_w_b, 40, stroke_color=ccolor, bg_color=ccolor, stroke_width=1, roundness={"type": 3}))
        elements.append(create_text(f"card_c_txt_{cx}", cx + 12, cy + 10, ctitle, font_size=12, color="#ffffff", font_family=1))
        
        # Render LaTeX formula
        l_svg = render_latex_to_svg(clatex, f"sec_c_card_{cx}", fontsize=13.5, color=ccolor)
        fid = register_svg(l_svg)
        elements.append(create_image_element(f"card_c_img_{cx}", cx + 15, cy + 50, col_w_b - 30, 48, fid, latex_source=clatex))
        
        # Bullets
        elements.append(create_text(f"card_c_bul_{cx}", cx + 18, cy + 110, cbullets, font_size=12, color="#e2e8f0", font_family=1, width=col_w_b - 36))

    # =========================================================================
    # SECTION D: 2024–2026 SOTA NORMALIZATION ADVANCEMENTS (y = 1930 .. 2550)
    # =========================================================================
    sec_d_box = create_rectangle("sec_d_frame", 40, 1930, 2420, 600, stroke_color="#f59e0b", bg_color="#181308", stroke_width=2)
    elements.append(sec_d_box)
    elements.append(create_text("sec_d_title", 60, 1948, "SECTION D: 2024–2026 CUTTING-EDGE INNOVATIONS (QK-NORM, ADALN-ZERO, DYT, DERF & HARDWARE KERNELS)", font_size=15, color="#fbbf24", font_family=1))

    # 3 Cards in Section D: QK-Norm, AdaLN-Zero (DiT), Dynamic Tanh & Derf
    d_cards = [
        ("D.1 QK-NORM: ATTENTION ENTROPY STABILIZER", 60, 1985, "#38bdf8",
         r"\mathbf{Q}' = \text{RMS}(\mathbf{Q}), \; \mathbf{K}' = \text{RMS}(\mathbf{K}), \quad \mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}'(\mathbf{K}')^T}{\sqrt{d_k}} \cdot s\right)",
         "• Mandatory standard for 2024-2026 LLMs (Qwen 2.5, OLMo 2, Kimi K2).\n• Solves Attention Logit Explosion in long-context (128k - 1M tokens).\n• Prevents Softmax saturation and attention sink entropy collapse.\n• Ensures numerical stability for low-precision FP8/FP16 training & edge inference."),

        ("D.2 ADALN-ZERO IN DIFFUSION TRANSFORMERS (DiT)", 880, 1985, "#10b981",
         r"\mathbf{y} = \boldsymbol{\alpha}_c \odot \mathcal{F}\left(\text{LN}(\mathbf{x}) \odot (1 + \boldsymbol{\gamma}_c) + \boldsymbol{\beta}_c\right), \quad \boldsymbol{\alpha}_c(0) = 0",
         "• Peebles & Xie (DiT - ICCV 2023 / Sora Foundation):\n  Modulates scale/shift/gate from timestep t and class/prompt c.\n• AdaLN-Zero initializes gate alpha = 0, making DiT start as pure Identity.\n• Eliminates need for custom warmups in generative diffusion architectures.\n• Enables massive parameter scaling in video and image generative models."),

        ("D.3 NORMALIZATION-FREE FRONTIERS: DYT & DERF", 1700, 1985, "#ec4899",
         r"\text{DyT}(\mathbf{x}) = \boldsymbol{\gamma} \odot \tanh(\boldsymbol{\alpha} \odot \mathbf{x}) + \boldsymbol{\beta}, \quad \text{Derf}(\mathbf{x}) = \text{erf}(\boldsymbol{\alpha} \odot \mathbf{x} + \mathbf{s})",
         "• Dynamic Tanh (CVPR 2025): Replaces statistical norm with element-wise tanh.\n• Derf (2026): Rescaled Gaussian error function matches LayerNorm expressivity.\n• Eliminates ALL cross-token communication and reduction passes.\n• Zero memory-bound synchronization: 20-30% faster execution on NPU/GPU.")
    ]

    for ctitle, cx, cy, ccolor, clatex, cbullets in d_cards:
        elements.append(create_rectangle(f"card_d_{cx}", cx, cy, col_w_b, 525, stroke_color=ccolor, bg_color="#0f172a", stroke_width=1.5))
        elements.append(create_rectangle(f"card_d_head_{cx}", cx, cy, col_w_b, 40, stroke_color=ccolor, bg_color=ccolor, stroke_width=1, roundness={"type": 3}))
        elements.append(create_text(f"card_d_txt_{cx}", cx + 12, cy + 10, ctitle, font_size=12, color="#ffffff", font_family=1))
        
        # Render LaTeX formula
        l_svg = render_latex_to_svg(clatex, f"sec_d_card_{cx}", fontsize=13.5, color=ccolor)
        fid = register_svg(l_svg)
        elements.append(create_image_element(f"card_d_img_{cx}", cx + 15, cy + 50, col_w_b - 30, 48, fid, latex_source=clatex))
        
        # Bullets
        elements.append(create_text(f"card_d_bul_{cx}", cx + 18, cy + 110, cbullets, font_size=12, color="#e2e8f0", font_family=1, width=col_w_b - 36))

    # Add Mini Illustrations inside D.1, D.2, D.3
    svg_d1 = render_vector_illustration(render_mini_qknorm, "sec_d_qknorm.svg", figsize=(4.0, 1.7), dpi=220)
    fid_d1 = register_svg(svg_d1)
    elements.append(create_image_element("sec_d1_img", 100, 2260, 700, 230, fid_d1))

    svg_d2 = render_vector_illustration(render_mini_adaln, "sec_d_adaln.svg", figsize=(4.0, 1.7), dpi=220)
    fid_d2 = register_svg(svg_d2)
    elements.append(create_image_element("sec_d2_img", 920, 2260, 700, 230, fid_d2))

    svg_d3 = render_vector_illustration(render_mini_dyt_derf, "sec_d_dyt_derf.svg", figsize=(4.0, 1.7), dpi=220)
    fid_d3 = register_svg(svg_d3)
    elements.append(create_image_element("sec_d3_img", 1740, 2260, 700, 230, fid_d3))

    # Save to Excalidraw JSON
    excalidraw_doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#020617",
            "gridSize": 20,
            "theme": "dark"
        },
        "files": files_dict
    }

    out_file = DIAGRAMS_DIR / "layernorm_evolution_2026.excalidraw"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(excalidraw_doc, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã tạo thành công Excalidraw Architecture Diagram: {out_file} ({len(elements)} elements, {len(files_dict)} SVGs)")

if __name__ == "__main__":
    build_layernorm_master_diagram()
