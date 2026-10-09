#!/usr/bin/env python3
"""
Generate Pure Geometric Illustrations for Deep Learning Normalization Foundations
==================================================================================
Produces 6 high-contrast, publication-grade geometric figures:
  1. geo_01_loss_contour_zigzag.png: Elongated Ellipsoid (Zig-zag) vs Concentric Circles (Direct Gradient)
  2. geo_02_feature_space_transformation.png: Raw Scattered -> Zero-Centered -> Unit-Sphere Isotropic
  3. geo_03_activation_saturation_zones.png: Sigmoid/Tanh Saturation Dead Zones vs Normalized Active Linear Band
  4. geo_04_tensor_slicing_3d.png: 3D Geometric Tensor Slices (BatchNorm, LayerNorm, InstanceNorm, GroupNorm)
  5. geo_05_weight_vs_spectral_norm.png: Direction-Magnitude Decoupling & Lipschitz 1-Bound Hypersphere Mapping
  6. geo_06_normalization_taxonomy_map.png: Global Geometric Taxonomy (Input Scaling, Activation Norm, Weight Norm)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "diagrams" / "normalization_geometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
BG_DARK = '#0b1329'
CARD_BG = '#0f172a'

def setup_fig(figsize=(9.5, 4.0), dpi=250):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    return fig, ax

def save_plot(fig, filename):
    out_path = OUTPUT_DIR / filename
    plt.tight_layout(pad=0.2)
    plt.savefig(str(out_path), format='png', dpi=250, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved Geometric Illustration: {out_path.name}")

# =============================================================================
# 1. HÌNH 1: LOSS CONTOURS & GRADIENT DESCENT (ZIG-ZAG VS DIRECT)
# =============================================================================
def render_geo_loss_contour():
    fig, ax = setup_fig((10.0, 4.2))
    
    # Title
    ax.text(0.5, 0.93, "HÌNH HỌC BỀ MẶT LOSS (LOSS CONTOURS) & ĐỘNG LỰC HỌC GRADIENT DESCENT", 
            color='#38bdf8', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

    # Left Frame: Unnormalized (Elongated Ellipse)
    ax.add_patch(patches.Rectangle((0.04, 0.08), 0.44, 0.78, fc='#1e293b', ec='#ef4444', lw=2, transform=ax.transAxes))
    ax.text(0.26, 0.80, "CHƯA CHUẨN HÓA (UNNORMALIZED)", color='#f87171', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.73, r"Thang đo lệch nhau: $x_1 \gg x_2 \to \kappa(\mathbf{H}) \gg 10^4$", color='#fca5a5', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Draw 4 nested ellipses
    cx, cy = 0.26, 0.40
    for w_fac, h_fac, col in [(0.32, 0.09, '#450a0a'), (0.24, 0.065, '#7f1d1d'), (0.16, 0.045, '#991b1b'), (0.08, 0.025, '#dc2626')]:
        ax.add_patch(patches.Ellipse((cx, cy), w_fac, h_fac, angle=35, fc=col, ec='#ef4444', lw=1.2, alpha=0.7, transform=ax.transAxes))
    
    # Target minimum w*
    ax.scatter([cx], [cy], color='#22c55e', s=60, zorder=5, transform=ax.transAxes)
    ax.text(cx + 0.02, cy - 0.04, "Optimum w*", color='#4ade80', fontsize=8, fontweight='bold', transform=ax.transAxes)

    # Zig-zag trajectory
    traj_x = [0.12, 0.22, 0.17, 0.29, 0.24, 0.26]
    traj_y = [0.22, 0.35, 0.48, 0.42, 0.41, 0.40]
    for i in range(len(traj_x)-1):
        ax.annotate('', xy=(traj_x[i+1], traj_y[i+1]), xytext=(traj_x[i], traj_y[i]), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", color='#fbbf24', lw=1.8))
    ax.text(0.12, 0.18, "Start w_0", color='#fbbf24', fontsize=7.5, transform=ax.transAxes)
    ax.text(0.26, 0.12, "Dao động Zig-zag dữ dội giữa 2 vách dốc\nGradient KHÔNG chỉ về tâm -> Hội tụ cực chậm", color='#fca5a5', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right Frame: Normalized (Concentric Circles)
    ax.add_patch(patches.Rectangle((0.52, 0.08), 0.44, 0.78, fc='#1e293b', ec='#10b981', lw=2, transform=ax.transAxes))
    ax.text(0.74, 0.80, "ĐÃ CHUẨN HÓA (STANDARDIZED)", color='#34d399', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.73, r"Thang đo đồng nhất: $\mu=0, \sigma=1 \to \kappa(\mathbf{H}) \approx 1$", color='#6ee7b7', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Draw 4 concentric circles
    rcx, rcy = 0.74, 0.42
    for r_rad, col in [(0.18, '#064e3b'), (0.13, '#065f46'), (0.085, '#047857'), (0.04, '#059669')]:
        ax.add_patch(patches.Circle((rcx, rcy), r_rad, fc=col, ec='#10b981', lw=1.2, alpha=0.7, transform=ax.transAxes))
    
    # Target minimum w*
    ax.scatter([rcx], [rcy], color='#22c55e', s=60, zorder=5, transform=ax.transAxes)
    ax.text(rcx + 0.02, rcy - 0.04, "Optimum w*", color='#4ade80', fontsize=8, fontweight='bold', transform=ax.transAxes)

    # Direct trajectory
    rtraj_x = [0.60, 0.67, 0.74]
    rtraj_y = [0.25, 0.33, 0.42]
    for i in range(len(rtraj_x)-1):
        ax.annotate('', xy=(rtraj_x[i+1], rtraj_y[i+1]), xytext=(rtraj_x[i], rtraj_y[i]), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2.2))
    ax.text(0.60, 0.21, "Start w_0", color='#38bdf8', fontsize=7.5, transform=ax.transAxes)
    ax.text(0.74, 0.12, "Gradient vuông góc đường mức = CHỈ THẲNG VỀ TÂM w*\nBước nhảy trơn tru -> Cho phép Learning Rate lớn", color='#a7f3d0', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_plot(fig, "geo_01_loss_contour_zigzag.png")

# =============================================================================
# 2. HÌNH 2: KHÔNG GIAN DỮ LIỆU BIẾN ĐỔI (FEATURE SPACE TRANSFORMATION)
# =============================================================================
def render_geo_feature_space():
    fig, ax = setup_fig((10.5, 3.8))
    
    ax.text(0.5, 0.93, "HÌNH HỌC BIẾN ĐỔI KHÔNG GIAN ĐẶC TRƯNG: RAW -> ZERO-CENTERED -> STANDARDIZED", 
            color='#10b981', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

    steps = [
        ("1. RAW DATA (Dữ liệu Thô)", 0.03, '#ef4444',
         "• Phân phối lệch tâm (Shifted Mean)\n• Biên độ dị hướng (Anisotropic Spread)\n• Trọng số bị chi phối bởi trục lớn nhất",
         (0.12, 0.46), 0.07, 0.03, 30),
        
        ("2. ZERO-CENTERED (Trừ Mean)", 0.36, '#38bdf8',
         "• Dịch tâm đám mây về gốc tọa độ (0, 0)\n• Triệt tiêu thành phần DC Offset\n• Cân bằng đối xứng qua các góc phần tư",
         (0.50, 0.45), 0.07, 0.03, 30),
        
        ("3. STANDARDIZED (Chia Std)", 0.69, '#a855f7',
         "• Co giãn về hình cầu đơn vị bán kính 1\n• Phương sai đồng nhất trên mọi chiều\n• Đẳng hướng hoàn hảo (Isotropic Sphere)",
         (0.83, 0.45), 0.045, 0.045, 0)
    ]

    for title, x_left, col, bullets, center, ew, eh, ang in steps:
        # Card Box
        ax.add_patch(patches.Rectangle((x_left, 0.08), 0.28, 0.78, fc='#1e293b', ec=col, lw=1.8, transform=ax.transAxes))
        ax.text(x_left + 0.14, 0.80, title, color=col, fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Coordinate axes
        ax_cx, ax_cy = x_left + 0.14, 0.45
        ax.plot([ax_cx - 0.09, ax_cx + 0.09], [ax_cy, ax_cy], color='#475569', lw=1, ls='--', transform=ax.transAxes)
        ax.plot([ax_cx, ax_cx], [ax_cy - 0.14, ax_cy + 0.14], color='#475569', lw=1, ls='--', transform=ax.transAxes)
        ax.scatter([ax_cx], [ax_cy], color='#94a3b8', s=15, transform=ax.transAxes) # origin (0,0)
        
        # Scatter cluster ellipse
        ax.add_patch(patches.Ellipse(center, ew, eh, angle=ang, fc=col, ec='#ffffff', lw=1.2, alpha=0.6, transform=ax.transAxes))
        
        # Bullets below
        ax.text(x_left + 0.14, 0.18, bullets, color='#cbd5e1', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Arrows between cards
    ax.annotate('', xy=(0.34, 0.45), xytext=(0.32, 0.45), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2.5))
    ax.text(0.33, 0.52, r"$-\mu$", color='#38bdf8', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)

    ax.annotate('', xy=(0.67, 0.45), xytext=(0.65, 0.45), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#a855f7', lw=2.5))
    ax.text(0.66, 0.52, r"$/\sigma$", color='#c084fc', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)

    save_plot(fig, "geo_02_feature_space_transformation.png")

# =============================================================================
# 3. HÌNH 3: VÙNG BÃO HÒA PHI TUYẾN TÍNH (ACTIVATION SATURATION ZONES)
# =============================================================================
def render_geo_activation_saturation():
    fig, ax = setup_fig((10.0, 4.0))
    
    ax.text(0.5, 0.93, "HÌNH HỌC VÙNG KÍCH HOẠT PHI TUYẾN: BÃO HÒA TRIỆT TIÊU GRADIENT vs VÙNG NHẠY TUYẾN TÍNH", 
            color='#f59e0b', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)

    # Create coordinate system inside frame
    ax.add_patch(patches.Rectangle((0.06, 0.08), 0.88, 0.78, fc='#1e293b', ec='#f59e0b', lw=2, transform=ax.transAxes))
    
    # Sigmoid / Tanh curve coordinates
    z = np.linspace(-6, 6, 200)
    sig = 1 / (1 + np.exp(-z))
    dsig = sig * (1 - sig) # Derivative
    
    # Subplot placement
    sub_ax = fig.add_axes([0.10, 0.16, 0.52, 0.65])
    sub_ax.set_facecolor('#0f172a')
    
    # Active Linear Zone rectangle [-2, 2]
    sub_ax.axvspan(-2, 2, color='#10b981', alpha=0.22, label='Active Linear Zone (Nhạy gradient)')
    
    # Dead Saturation Zones (<-2 and >2)
    sub_ax.axvspan(-6, -2, color='#ef4444', alpha=0.18, label="Dead Zone Trai (f'(z) -> 0)")
    sub_ax.axvspan(2, 6, color='#ef4444', alpha=0.18, label="Dead Zone Phai (f'(z) -> 0)")

    # Plot curves
    sub_ax.plot(z, sig, color='#38bdf8', lw=2.5, label='Sigmoid f(z)')
    sub_ax.plot(z, dsig, color='#f59e0b', lw=2.0, ls='--', label="Gradient f'(z)")
    sub_ax.axhline(0, color='#475569', lw=1, ls=':')
    sub_ax.axvline(0, color='#475569', lw=1, ls=':')

    sub_ax.set_xlim(-6, 6)
    sub_ax.set_ylim(-0.05, 1.05)
    sub_ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in sub_ax.spines.values():
        spine.set_color('#475569')

    # Right annotations inside main figure
    ax.text(0.67, 0.74, "TẠI SAO PHẢI CHUẨN HÓA?", color='#fbbf24', fontsize=9.5, fontweight='bold', transform=ax.transAxes)
    ax.text(0.67, 0.54, "1. KHÔNG CHUẨN HÓA:\nKích hoạt |z| > 4 bị trôi vào Dead Zone.\nĐạo hàm f'(z) = 0 triệt tiêu hoàn toàn\nlan truyền ngược (Vanishing Gradient).", color='#fca5a5', fontsize=8, transform=ax.transAxes)
    ax.text(0.67, 0.28, "2. CÓ CHUẨN HÓA (Normalized):\nKhống chế z ~ N(0, 1) nằm trọn trong [-2, +2].\nĐạo hàm duy trì mức tối đa f'(0) = 0.25,\ngradient truyền mượt mà qua hàng trăm tầng.", color='#6ee7b7', fontsize=8, transform=ax.transAxes)

    out_path = OUTPUT_DIR / "geo_03_activation_saturation_zones.png"
    plt.savefig(str(out_path), format='png', dpi=250, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved Geometric Illustration: {out_path.name}")

# =============================================================================
# 4. HÌNH 4: 3D TENSOR SLICING GEOMETRY (BN, LN, IN, GN)
# =============================================================================
def render_geo_tensor_slicing():
    fig, ax = setup_fig((10.5, 4.0))
    
    ax.text(0.5, 0.93, "HÌNH HỌC LÁT CẮT TENSOR 3D (TENSOR SLICING GEOMETRY): BATCHNORM, LAYERNORM, INSTANCE, GROUP", 
            color='#a855f7', fontsize=11.5, fontweight='bold', ha='center', transform=ax.transAxes)

    cards = [
        ("BATCH NORM (BN)", 0.03, '#3b82f6',
         "• Cắt dọc trục Batch (N)\n• Chuẩn hóa 1 kênh trên toàn bộ batch mẫu\n• Thất bại khi Batch < 32 / Chuỗi động"),
        
        ("LAYER NORM (LN)", 0.27, '#10b981',
         "• Cắt ngang trục Kênh (C)\n• Chuẩn hóa toàn bộ kênh của 1 mẫu độc lập\n• Chuẩn mực Transformers / ViT / LLMs"),
        
        ("INSTANCE NORM (IN)", 0.51, '#f59e0b',
         "• Cắt từng ô đơn lẻ (1 Mẫu, 1 Kênh)\n• Loại bỏ tương phản màu/phong cách\n• Chuẩn mực Style Transfer / GANs"),
        
        ("GROUP NORM (GN)", 0.75, '#ec4899',
         "• Chia C thành G nhóm kênh độc lập\n• Trung gian dung hòa giữa BN và LN\n• Tối ưu thị giác máy tính với Batch nhỏ")
    ]

    for title, x_left, col, bullets in cards:
        ax.add_patch(patches.Rectangle((x_left, 0.08), 0.22, 0.78, fc='#1e293b', ec=col, lw=1.8, transform=ax.transAxes))
        ax.text(x_left + 0.11, 0.80, title, color=col, fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
        
        # 3D Cube visual representation
        cx, cy = x_left + 0.04, 0.40
        # Front face (N x C)
        ax.add_patch(patches.Rectangle((cx, cy), 0.08, 0.16, fc='#334155', ec='#64748b', lw=1, alpha=0.5, transform=ax.transAxes))
        # Top face
        top_poly = np.array([[cx, cy + 0.16], [cx + 0.04, cy + 0.22], [cx + 0.12, cy + 0.22], [cx + 0.08, cy + 0.16]])
        ax.add_patch(patches.Polygon(top_poly, fc='#475569', ec='#64748b', lw=1, alpha=0.5, transform=ax.transAxes))
        # Right face
        r_poly = np.array([[cx + 0.08, cy], [cx + 0.12, cy + 0.06], [cx + 0.12, cy + 0.22], [cx + 0.08, cy + 0.16]])
        ax.add_patch(patches.Polygon(r_poly, fc='#1e293b', ec='#64748b', lw=1, alpha=0.5, transform=ax.transAxes))

        # Highlight Slice based on Norm Type
        if "BATCH" in title: # Vertical slice across N
            ax.add_patch(patches.Rectangle((cx, cy), 0.08, 0.04, fc=col, ec='#ffffff', lw=1.2, alpha=0.9, transform=ax.transAxes))
        elif "LAYER" in title: # Horizontal slice across C
            ax.add_patch(patches.Rectangle((cx, cy), 0.02, 0.16, fc=col, ec='#ffffff', lw=1.2, alpha=0.9, transform=ax.transAxes))
        elif "INSTANCE" in title: # Single element
            ax.add_patch(patches.Rectangle((cx, cy), 0.02, 0.04, fc=col, ec='#ffffff', lw=1.2, alpha=0.9, transform=ax.transAxes))
        elif "GROUP" in title: # Sub-group of channels
            ax.add_patch(patches.Rectangle((cx, cy), 0.04, 0.16, fc=col, ec='#ffffff', lw=1.2, alpha=0.9, transform=ax.transAxes))

        # Bullets
        ax.text(x_left + 0.11, 0.20, bullets, color='#cbd5e1', fontsize=7.2, ha='center', transform=ax.transAxes)

    save_plot(fig, "geo_04_tensor_slicing_3d.png")

# =============================================================================
# 5. HÌNH 5: WEIGHT NORM & SPECTRAL NORM (LIPSCHITZ BOUNDS)
# =============================================================================
def render_geo_weight_spectral():
    fig, ax = setup_fig((10.0, 4.0))
    
    ax.text(0.5, 0.93, "HÌNH HỌC CHUẨN HÓA TRỌNG SỐ: WEIGHT NORMALIZATION vs SPECTRAL NORMALIZATION (LIPSCHITZ 1-BOUND)", 
            color='#38bdf8', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)

    # Left: Weight Normalization
    ax.add_patch(patches.Rectangle((0.05, 0.08), 0.43, 0.78, fc='#1e293b', ec='#38bdf8', lw=2, transform=ax.transAxes))
    ax.text(0.265, 0.80, "WEIGHT NORMALIZATION (Salimans 2016)", color='#38bdf8', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.265, 0.73, r"Phân rã: $\mathbf{w} = g \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$ (Độ lớn $g$ + Hướng $\hat{\mathbf{v}}$)", color='#bae6fd', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Draw vector decomposition
    ax.plot([0.15, 0.35], [0.38, 0.38], color='#475569', lw=1, ls='--', transform=ax.transAxes)
    ax.plot([0.25, 0.25], [0.28, 0.48], color='#475569', lw=1, ls='--', transform=ax.transAxes)
    # Unit sphere
    ax.add_patch(patches.Circle((0.25, 0.38), 0.08, fc='none', ec='#38bdf8', lw=1.2, ls=':', transform=ax.transAxes))
    # Direction vector v / ||v||
    ax.annotate('', xy=(0.31, 0.44), xytext=(0.25, 0.38), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#10b981', lw=2.2))
    ax.text(0.32, 0.45, r"$\hat{\mathbf{v}} \in \mathbb{S}^{d-1}$", color='#34d399', fontsize=8, transform=ax.transAxes)
    # Full vector w = g * v
    ax.annotate('', xy=(0.36, 0.49), xytext=(0.25, 0.38), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#f59e0b', lw=2.5))
    ax.text(0.37, 0.50, r"$\mathbf{w} = g\hat{\mathbf{v}}$", color='#fbbf24', fontsize=8.5, fontweight='bold', transform=ax.transAxes)
    ax.text(0.265, 0.16, "Tách rời tốc độ cập nhật góc xoay và độ dài\nGiúp tối ưu hóa không bị tắc nghẽn biên độ", color='#cbd5e1', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: Spectral Normalization
    ax.add_patch(patches.Rectangle((0.52, 0.08), 0.43, 0.78, fc='#1e293b', ec='#a855f7', lw=2, transform=ax.transAxes))
    ax.text(0.735, 0.80, "SPECTRAL NORMALIZATION (Miyato 2018)", color='#c084fc', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.735, 0.73, r"Khống chế giá trị kỳ dị: $\mathbf{W}_{\text{SN}} = \frac{\mathbf{W}}{\sigma_{\max}(\mathbf{W})}$", color='#e9d5ff', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Draw Matrix transformation mapping Sphere to Ellipsoid with max radius 1
    ax.add_patch(patches.Circle((0.62, 0.38), 0.06, fc='#0f172a', ec='#38bdf8', lw=1.2, transform=ax.transAxes))
    ax.text(0.62, 0.38, r"$\|\mathbf{x}\| \leq 1$", color='#38bdf8', fontsize=7.5, ha='center', va='center', transform=ax.transAxes)
    
    ax.annotate('', xy=(0.74, 0.38), xytext=(0.69, 0.38), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#c084fc', lw=2))
    ax.text(0.715, 0.42, r"$\mathbf{W}_{\text{SN}}$", color='#c084fc', fontsize=8.5, ha='center', transform=ax.transAxes)

    ax.add_patch(patches.Ellipse((0.83, 0.38), 0.14, 0.07, angle=25, fc='#581c87', ec='#a855f7', lw=1.5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.83, 0.38, r"$\sigma_{\max} = 1$", color='#ffffff', fontsize=7.5, ha='center', va='center', fontweight='bold', transform=ax.transAxes)

    ax.text(0.735, 0.16, "Ép hằng số Lipschitz toàn cục L <= 1\nNgăn chặn bùng nổ Gradient trong Discriminator/GANs", color='#cbd5e1', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_plot(fig, "geo_05_weight_vs_spectral_norm.png")

# =============================================================================
# 6. HÌNH 6: TOÀN CẢNH HỆ THỐNG CHUẨN HÓA (GLOBAL TAXONOMY MAP)
# =============================================================================
def render_geo_taxonomy_map():
    fig, ax = setup_fig((10.5, 4.2))
    
    ax.text(0.5, 0.93, "BẢN ĐỒ TOÀN CẢNH HỆ THỐNG CHUẨN HÓA TRONG HỌC SÂU (TAXONOMY LANDSCAPE)", 
            color='#10b981', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

    columns = [
        ("1. INPUT FEATURE SCALING", 0.03, '#3b82f6',
         ["• Min-Max Normalization: [0, 1]",
          "• Z-Score Standardization: N(0, 1)",
          "• Robust IQR Scaling (Chống Outlier)",
          "• L2 Unit Sphere Normalization"]),
        
        ("2. ACTIVATION NORMALIZATION", 0.36, '#10b981',
         ["• BatchNorm (Trục Batch B)",
          "• LayerNorm (Trục Feature d)",
          "• RMSNorm (Root Mean Square)",
          "• GroupNorm / InstanceNorm",
          "• AdaLN-Zero (Diffusion DiT)"]),
        
        ("3. PARAMETER / WEIGHT NORM", 0.69, '#a855f7',
         ["• Weight Normalization (g * v/||v||)",
          "• Spectral Normalization (Lipschitz 1)",
          "• Weight Standardization (WS)",
          "• Gain-Free / Bias-Free Norms",
          "• QK-Norm (Attention Stabilizer)"])
    ]

    for title, x_left, col, items in columns:
        ax.add_patch(patches.Rectangle((x_left, 0.08), 0.28, 0.78, fc='#1e293b', ec=col, lw=2, transform=ax.transAxes))
        ax.text(x_left + 0.14, 0.80, title, color=col, fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
        
        for idx, item in enumerate(items):
            ax.text(x_left + 0.02, 0.65 - idx * 0.11, item, color='#f8fafc', fontsize=8, transform=ax.transAxes)

    save_plot(fig, "geo_06_normalization_taxonomy_map.png")

def main():
    print("🚀 Đang khởi tạo bộ 6 hình học minh họa trực quan chuyên sâu về Normalization...")
    render_geo_loss_contour()
    render_geo_feature_space()
    render_geo_activation_saturation()
    render_geo_tensor_slicing()
    render_geo_weight_spectral()
    render_geo_taxonomy_map()
    print("✅ Hoàn tất render toàn bộ 6 hình học trực quan!")

if __name__ == "__main__":
    main()
