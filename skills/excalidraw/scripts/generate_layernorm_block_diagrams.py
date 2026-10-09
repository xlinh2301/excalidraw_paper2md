#!/usr/bin/env python3
"""
Generate Individual Visual Block Diagrams for Layer Normalization Foundation & 2024-2026 Survey
================================================================================================
Renders 17 modular, high-contrast, publication-grade PNG diagrams into docs/diagrams/layernorm_blocks/
Each block is designed for direct embedding into docs/research/LAYERNORM_FOUNDATION_AND_2024_2026_SURVEY.md
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "diagrams" / "layernorm_blocks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common styling
BG_DARK = '#0b1329'
CARD_BG = '#0f172a'

def setup_ax(figsize=(8.5, 3.0), dpi=250):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')
    return fig, ax

def save_and_close(fig, filename):
    out_path = OUTPUT_DIR / filename
    plt.tight_layout(pad=0.2)
    plt.savefig(str(out_path), format='png', dpi=250, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved: {out_path.name}")

# ==========================================
# 1. BATCHNORM VS LAYERNORM
# ==========================================
def render_bn_vs_ln():
    fig, ax = setup_ax((9.0, 3.2))
    
    # Left: BatchNorm
    ax.add_patch(patches.Rectangle((0.05, 0.1), 0.42, 0.8, fc='#1e293b', ec='#3b82f6', lw=2, transform=ax.transAxes))
    ax.text(0.26, 0.82, "BATCH NORMALIZATION (BN)", color='#60a5fa', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.65, r"$\mu_B = \frac{1}{|B|}\sum_{i \in B} x_i, \quad \sigma_B^2 = \frac{1}{|B|}\sum_{i \in B}(x_i - \mu_B)^2$", color='#93c5fd', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.46, "• Chuẩn hóa dọc theo chiều Batch (B)\n• Phụ thuộc kích thước Batch (B < 32 bị lỗi)\n• Thất bại trên chuỗi độ dài động / LLMs", color='#cbd5e1', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.add_patch(patches.Rectangle((0.08, 0.15), 0.36, 0.13, fc='#1d4ed8', ec='#93c5fd', lw=1, transform=ax.transAxes))
    ax.text(0.26, 0.21, "Batch-Dependent (Train != Eval)", color='#ffffff', fontsize=8.5, ha='center', fontweight='bold', transform=ax.transAxes)

    # Right: LayerNorm
    ax.add_patch(patches.Rectangle((0.53, 0.1), 0.42, 0.8, fc='#1e293b', ec='#10b981', lw=2, transform=ax.transAxes))
    ax.text(0.74, 0.82, "LAYER NORMALIZATION (LN)", color='#34d399', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.65, r"$\mu_L = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma_L^2 = \frac{1}{d}\sum_{i=1}^d(x_i - \mu_L)^2$", color='#6ee7b7', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.46, "• Chuẩn hóa độc lập trên chiều Feature (d)\n• Hoàn toàn độc lập với kích thước Batch\n• Đồng nhất tuyệt đối giữa Train và Inference", color='#cbd5e1', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.add_patch(patches.Rectangle((0.56, 0.15), 0.36, 0.13, fc='#047857', ec='#6ee7b7', lw=1, transform=ax.transAxes))
    ax.text(0.74, 0.21, "Batch-Agnostic (Chuẩn hóa từng Token)", color='#ffffff', fontsize=8.5, ha='center', fontweight='bold', transform=ax.transAxes)

    save_and_close(fig, "01_bn_vs_ln.png")

# ==========================================
# 2. BƯỚC 1: TÍNH MEAN μ
# ==========================================
def render_step1_mean():
    fig, ax = setup_ax((9.0, 2.7))
    xs = np.linspace(0.08, 0.62, 6)
    vals = [2.4, -1.2, 3.8, 0.5, -2.1, 4.0]
    ax.text(0.35, 0.85, "BƯỚC 1: TÍNH KỲ VỌNG TRUNG BÌNH (ACTIVATION MEAN)", color='#38bdf8', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    for i, (x_pos, val) in enumerate(zip(xs, vals)):
        lbl = f"x_{i+1}" if i < 5 else "x_d"
        ax.add_patch(patches.Rectangle((x_pos, 0.38), 0.08, 0.26, fc='#1e293b', ec='#38bdf8', lw=1.5, transform=ax.transAxes))
        ax.text(x_pos + 0.04, 0.51, f"{val:+.1f}", color='#ffffff', fontsize=9, ha='center', va='center', transform=ax.transAxes)
        ax.text(x_pos + 0.04, 0.26, f"${lbl}$", color='#94a3b8', fontsize=8.5, ha='center', transform=ax.transAxes)
    
    # Arrow to mean
    ax.annotate('', xy=(0.76, 0.51), xytext=(0.71, 0.51), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2.5))
    
    # Result Mean box
    ax.add_patch(patches.Rectangle((0.77, 0.28), 0.18, 0.46, fc='#0284c7', ec='#38bdf8', lw=2, transform=ax.transAxes))
    ax.text(0.86, 0.58, r"$\mu = \frac{1}{d}\sum_{i=1}^d x_i$", color='#ffffff', fontsize=10.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.86, 0.38, r"$\mu = +1.23$", color='#e0f2fe', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.86, 0.16, "DC Component (Thành phần dịch)", color='#7dd3fc', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "02_step1_mean.png")

# ==========================================
# 3. BƯỚC 2: TÍNH VARIANCE σ²
# ==========================================
def render_step2_var():
    fig, ax = setup_ax((9.0, 2.7))
    ax.text(0.5, 0.86, "BƯỚC 2: TÍNH PHƯƠNG SAI & ĐỘ LỆCH CHUẨN", color='#10b981', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Deviation from Mean
    ax.add_patch(patches.Rectangle((0.08, 0.22), 0.38, 0.52, fc='#1e293b', ec='#10b981', lw=1.5, transform=ax.transAxes))
    ax.text(0.27, 0.58, r"Độ lệch tâm: $(x_i - \mu)$", color='#34d399', fontsize=10.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.27, 0.36, "Đo lường mức độ phân tán của từng chiều\nxung quanh giá trị trung bình $\mu$", color='#cbd5e1', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Arrow
    ax.annotate('', xy=(0.51, 0.48), xytext=(0.47, 0.48), xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='#10b981', lw=2.5))

    # Right: Variance & Std
    ax.add_patch(patches.Rectangle((0.52, 0.18), 0.40, 0.58, fc='#047857', ec='#6ee7b7', lw=2, transform=ax.transAxes))
    ax.text(0.72, 0.58, r"$\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$", color='#ffffff', fontsize=10.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.72, 0.38, r"$\sigma = \sqrt{\sigma^2 + \epsilon} \quad (\epsilon \sim 10^{-5})$", color='#a7f3d0', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.72, 0.22, r"$\epsilon$: Hằng số chống chia cho 0", color='#d1fae5', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "03_step2_var.png")

# ==========================================
# 4. BƯỚC 3: CHUẨN HÓA STANDARDIZATION
# ==========================================
def render_step3_standardize():
    fig, ax = setup_ax((9.0, 2.8))
    ax.text(0.5, 0.86, "BƯỚC 3: CHUẨN HÓA ĐIỂM CHUẨN (STANDARDIZATION)", color='#a855f7', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Equation Box
    ax.add_patch(patches.Rectangle((0.06, 0.20), 0.48, 0.56, fc='#1e293b', ec='#a855f7', lw=2, transform=ax.transAxes))
    ax.text(0.30, 0.60, r"$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu \mathbf{1}}{\sqrt{\sigma^2 + \epsilon}} = \frac{\mathbf{x} - \mu \mathbf{1}}{\sigma}$", color='#e9d5ff', fontsize=11, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.30, 0.34, r"• $\sum_{i=1}^d \hat{x}_i = 0$  (Kỳ vọng trung bình bằng 0)" + "\n" + r"• $\frac{1}{d}\sum_{i=1}^d \hat{x}_i^2 \approx 1$  (Phương sai bằng 1)", color='#c084fc', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Geometry visual
    ax.add_patch(patches.Rectangle((0.57, 0.18), 0.37, 0.60, fc='#581c87', ec='#c084fc', lw=1.5, transform=ax.transAxes))
    ax.text(0.755, 0.64, "Ý NGHĨA HÌNH HỌC", color='#ffffff', fontsize=9.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.755, 0.44, r"Chiếu vector $\mathbf{x} \in \mathbb{R}^d$ lên Mặt cầu" + "\n" + r"đơn vị $\mathbb{S}^{d-1}$ trực giao với $\mathbf{1}$", color='#f3e8ff', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.text(0.755, 0.24, "Triệt tiêu 2 bậc tự do (dịch + co giãn)", color='#d8b4fe', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "04_step3_standardize.png")

# ==========================================
# 5. BƯỚC 4: AFFINE TRANSFORM
# ==========================================
def render_step4_affine():
    fig, ax = setup_ax((9.0, 2.8))
    ax.text(0.5, 0.86, "BƯỚC 4: ÁNH XẠ TUYẾN TÍNH HỌC ĐƯỢC (AFFINE TRANSFORMATION)", color='#f59e0b', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Equation Box
    ax.add_patch(patches.Rectangle((0.06, 0.18), 0.48, 0.58, fc='#1e293b', ec='#f59e0b', lw=2, transform=ax.transAxes))
    ax.text(0.30, 0.58, r"$\mathbf{y} = \text{LN}(\mathbf{x}) = \gamma \odot \hat{\mathbf{x}} + \beta$", color='#fde68a', fontsize=11.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.30, 0.33, r"• $\gamma \in \mathbb{R}^d$: Tham số co giãn (Scale/Gain, Init = $1$)" + "\n" + r"• $\beta \in \mathbb{R}^d$: Tham số dịch chuyển (Shift/Bias, Init = $0$)", color='#fcd34d', fontsize=8.5, ha='center', transform=ax.transAxes)

    # Rationale Box
    ax.add_patch(patches.Rectangle((0.57, 0.18), 0.37, 0.58, fc='#78350f', ec='#fbbf24', lw=1.5, transform=ax.transAxes))
    ax.text(0.755, 0.64, "TẠI SAO CẦN AFFINE?", color='#ffffff', fontsize=9.5, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.755, 0.44, "Khôi phục sức chứa biểu diễn (Expressive Power):\nCho phép mạng tự học cách dịch hoặc giãn lại\nnếu chuẩn hóa thuần túy làm mất tín hiệu.", color='#fef3c7', fontsize=8, ha='center', transform=ax.transAxes)
    ax.text(0.755, 0.24, r"Nếu $\gamma=\sigma, \beta=\mu \to \mathbf{y}=\mathbf{x}$ (Identity Mapping)", color='#fde68a', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "05_step4_affine.png")

# ==========================================
# 6. INVARIANCE LAWS
# ==========================================
def render_invariance_laws():
    fig, ax = setup_ax((9.2, 3.0))
    ax.text(0.5, 0.88, "3 ĐỊNH LUẬT BẤT BIẾN TOÁN HỌC & CƠ CHẾ TỰ ĐIỀU HÒA GRADIENT", color='#38bdf8', fontsize=11.5, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # 3 sub-boxes
    boxes = [
        ("1. SHIFT INVARIANCE", r"$\text{LN}(\mathbf{x} + c\mathbf{1}) = \text{LN}(\mathbf{x})$", "Loại bỏ hiện tượng trôi dạt\ngiá trị DC qua các tầng sâu.", 0.05, '#3b82f6'),
        ("2. SCALE INVARIANCE", r"$\text{LN}(\alpha \mathbf{x}) = \text{LN}(\mathbf{x})$", "Đầu ra không bị ảnh hưởng bởi\nbiên độ kích hoạt tăng vọt.", 0.37, '#10b981'),
        ("3. WEIGHT SCALING & GRAD", r"$\frac{\partial \mathcal{L}}{\partial (\alpha \mathbf{W})} = \frac{1}{\alpha}\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$", "Trọng số càng lớn thì gradient\ntự động co nhỏ (Auto-Regulate).", 0.69, '#a855f7')
    ]
    for title, formula, desc, x_pos, col in boxes:
        ax.add_patch(patches.Rectangle((x_pos, 0.15), 0.27, 0.65, fc='#1e293b', ec=col, lw=1.5, transform=ax.transAxes))
        ax.text(x_pos + 0.135, 0.68, title, color=col, fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(x_pos + 0.135, 0.48, formula, color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(x_pos + 0.135, 0.26, desc, color='#cbd5e1', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "06_invariance_laws.png")

# ==========================================
# 7. JACOBIAN MATRIX & S^{d-1} GEOMETRY
# ==========================================
def render_jacobian_geometry():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "MA TRẬN JACOBIAN & HÌNH HỌC VI PHÂN TRÊN ĐA TẠP MẶT CẦU", color='#10b981', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Jacobian formula
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.46, 0.65, fc='#1e293b', ec='#10b981', lw=1.5, transform=ax.transAxes))
    ax.text(0.28, 0.68, "MA TRẬN HÌNH CHIẾU TRỰC GIAO", color='#34d399', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.28, 0.50, r"$\mathbf{J}_{\text{LN}} = \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} = \frac{1}{\sigma}\left(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^T\right)$", color='#ffffff', fontsize=8.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.28, 0.32, r"$\mathbf{J}_{\text{LN}} = \frac{1}{\sigma}\mathbf{P}_{\hat{\mathbf{x}}}^\perp \mathbf{P}_{\mathbf{1}}^\perp$", color='#a7f3d0', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.28, 0.20, "Tích của 2 toán tử chiếu trực giao liên tiếp", color='#6ee7b7', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: Geometric flow
    ax.add_patch(patches.Rectangle((0.54, 0.15), 0.41, 0.65, fc='#064e3b', ec='#34d399', lw=1.5, transform=ax.transAxes))
    ax.text(0.745, 0.68, "ĐỘNG LỰC HỌC TIẾP TUYẾN", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.745, 0.48, r"$\langle \frac{\partial \mathcal{L}}{\partial \mathbf{x}}, \mathbf{1} \rangle = 0, \quad \langle \frac{\partial \mathcal{L}}{\partial \mathbf{x}}, \hat{\mathbf{x}} \rangle = 0$", color='#a7f3d0', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.745, 0.28, "Mọi gradient luôn trượt trên KHÔNG GIAN TIẾP TUYẾN\ncủa mặt cầu, triệt tiêu hoàn toàn dao động biên độ.", color='#d1fae5', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "07_jacobian_geometry.png")

# ==========================================
# 8. LOSS LANDSCAPE SMOOTHING
# ==========================================
def render_loss_landscape():
    fig, ax = setup_ax((8.8, 2.8))
    ax.text(0.5, 0.88, "LÀM MỊN BỀ MẶT HÀM MẤT MÁT (LOSS LANDSCAPE) & HESSIAN CONDITIONING", color='#a855f7', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Without Norm
    ax.add_patch(patches.Rectangle((0.08, 0.15), 0.38, 0.65, fc='#1e293b', ec='#ef4444', lw=1.5, transform=ax.transAxes))
    ax.text(0.27, 0.68, "KHÔNG CÓ LAYERNORM", color='#f87171', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.27, 0.48, r"$\kappa(\mathbf{H}) = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 10^4$", color='#fca5a5', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.27, 0.28, "Bề mặt dạng khe núi hẹp (Ravines),\ngradient dao động mạnh, dễ nổ/phân kỳ.", color='#fecaca', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: With Norm
    ax.add_patch(patches.Rectangle((0.54, 0.15), 0.38, 0.65, fc='#1e293b', ec='#10b981', lw=1.5, transform=ax.transAxes))
    ax.text(0.73, 0.68, "CÓ LAYERNORM (SMOOTH BOWL)", color='#34d399', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.73, 0.48, r"$\kappa(\mathbf{H}) \approx 1, \quad L_{\text{smooth}} \ll L_{\text{raw}}$", color='#6ee7b7', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.73, 0.28, "Hàm mất mát Lipschitz liên tục,\ncho phép Learning Rate lớn hơn gấp 5–10 lần.", color='#a7f3d0', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "08_loss_landscape.png")

# ==========================================
# 9. POST-LN VS PRE-LN
# ==========================================
def render_post_vs_pre_ln():
    fig, ax = setup_ax((9.0, 3.2))
    ax.text(0.5, 0.88, "SO SÁNH TIẾN HÓA VỊ TRÍ: POST-LN (2017) vs PRE-LN (2020)", color='#38bdf8', fontsize=11.5, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Post-LN
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.42, 0.65, fc='#1e293b', ec='#ef4444', lw=1.5, transform=ax.transAxes))
    ax.text(0.26, 0.68, "POST-LN (Vaswani et al., 2017)", color='#f87171', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.50, r"$\mathbf{x}_l = \text{LN}(\mathbf{x}_{l-1} + \mathcal{F}(\mathbf{x}_{l-1}))$", color='#ffffff', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.32, r"Gradient tầng sâu: $\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_0} \sim \mathcal{O}(L^{-1})$", color='#fca5a5', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.text(0.26, 0.20, "Bắt buộc phải có Warm-up Learning Rate", color='#f87171', fontsize=7.5, ha='center', fontweight='bold', transform=ax.transAxes)

    # Right: Pre-LN
    ax.add_patch(patches.Rectangle((0.53, 0.15), 0.42, 0.65, fc='#1e293b', ec='#10b981', lw=1.5, transform=ax.transAxes))
    ax.text(0.74, 0.68, "PRE-LN (Xiong et al., 2020)", color='#34d399', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.50, r"$\mathbf{x}_l = \mathbf{x}_{l-1} + \mathcal{F}(\text{LN}(\mathbf{x}_{l-1}))$", color='#ffffff', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.32, r"Đường cao tốc: $\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \mathbf{I} + \sum \frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_l}$", color='#6ee7b7', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.20, "Identity Highway: Huấn luyện ổn định tức thì", color='#10b981', fontsize=7.5, ha='center', fontweight='bold', transform=ax.transAxes)

    save_and_close(fig, "09_post_vs_pre_ln.png")

# ==========================================
# 10. DEEPNORM SCALING
# ==========================================
def render_deepnorm():
    fig, ax = setup_ax((8.8, 2.8))
    ax.text(0.5, 0.88, "DEEPNORM (MICROSOFT DEEPNET) — MỞ RỘNG MẠNG ĐẾN 1,000 TẦNG", color='#38bdf8', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.add_patch(patches.Rectangle((0.08, 0.15), 0.84, 0.65, fc='#1e293b', ec='#38bdf8', lw=2, transform=ax.transAxes))
    ax.text(0.5, 0.66, r"$\mathbf{x}_l = \text{LN}\left(\mathbf{x}_{l-1} \cdot \alpha + \mathcal{F}(\mathbf{x}_{l-1}, \mathbf{W}_l \cdot \beta)\right)$", color='#38bdf8', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.44, r"Hệ số lý thuyết: $\alpha = (2L)^{1/4}, \quad \beta = (8L)^{-1/4}$", color='#e0f2fe', fontsize=10.5, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, r"Khống chế phương sai tích lũy: $\text{Var}(\mathbf{x}_L) = \Theta(1)$ với mọi độ sâu $L$ (Không bị bùng nổ kích hoạt).", color='#7dd3fc', fontsize=8.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "10_deepnorm.png")

# ==========================================
# 11. PARALLEL BLOCKS
# ==========================================
def render_parallel_blocks():
    fig, ax = setup_ax((8.8, 2.8))
    ax.text(0.5, 0.88, "PARALLEL TRANSFORMER BLOCKS (PaLM / GPT-J / LLaMA-PARALLEL)", color='#f59e0b', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.add_patch(patches.Rectangle((0.08, 0.15), 0.84, 0.65, fc='#1e293b', ec='#f59e0b', lw=2, transform=ax.transAxes))
    ax.text(0.5, 0.66, r"$\mathbf{x}_l = \mathbf{x}_{l-1} + \text{Attn}(\text{LN}(\mathbf{x}_{l-1})) + \text{MLP}(\text{LN}(\mathbf{x}_{l-1}))$", color='#fbbf24', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.44, "Shared LayerNorm: Chạy Attention và MLP song song cùng lúc", color='#ffffff', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, "Fuses $W_q, W_k, W_v$ và $W_{\text{gate}}, W_{\text{up}}$ thành 1 ma trận GEMM duy nhất -> Tăng 15% thông lượng GPU.", color='#fde68a', fontsize=8.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "11_parallel_blocks.png")

# ==========================================
# 12. RMSNORM MECHANISM
# ==========================================
def render_rmsnorm():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "RMSNORM — CHUẨN CÔNG NGHIỆP LLM (LLaMA 3, Gemma 2, Qwen 2.5, DeepSeek-V3)", color='#a855f7', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Formula
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.45, 0.65, fc='#1e293b', ec='#a855f7', lw=1.5, transform=ax.transAxes))
    ax.text(0.275, 0.68, "CÔNG THỨC TOÁN HỌC", color='#c084fc', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.50, r"$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$", color='#ffffff', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.32, r"$\mathbf{y} = \text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$", color='#e9d5ff', fontsize=10, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.20, r"Loại bỏ hoàn toàn phép trừ $\mu$ và Bias $\beta$", color='#d8b4fe', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: GPU speedup
    ax.add_patch(patches.Rectangle((0.53, 0.15), 0.42, 0.65, fc='#4c1d95', ec='#c084fc', lw=1.5, transform=ax.transAxes))
    ax.text(0.74, 0.68, "TỐI ƯU BĂNG THÔNG GPU", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.50, "LayerNorm: 2 lần quét bộ nhớ (Mean, Var)\nRMSNorm: CHỈ 1 LẦN QUÉT DUY NHẤT", color='#e9d5ff', fontsize=8.5, ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.26, "Tăng 25% - 50% tốc độ thực thi kernel Triton/CUDA!", color='#a7f3d0', fontsize=8.5, ha='center', fontweight='bold', transform=ax.transAxes)

    save_and_close(fig, "12_rmsnorm.png")

# ==========================================
# 13. QK-NORM
# ==========================================
def render_qknorm():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "QK-NORM (QUERY-KEY NORMALIZATION) — CHỐNG ATTENTION LOGIT EXPLOSION", color='#38bdf8', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: Formula
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.45, 0.65, fc='#1e293b', ec='#38bdf8', lw=1.5, transform=ax.transAxes))
    ax.text(0.275, 0.68, "CƠ CHẾ CHUẨN HÓA Q & K", color='#38bdf8', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.50, r"$\mathbf{Q}' = \text{RMSNorm}(\mathbf{Q}), \quad \mathbf{K}' = \text{RMSNorm}(\mathbf{K})$", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.32, r"$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}'(\mathbf{K}')^T}{\sqrt{d_k}} \cdot s\right) \mathbf{V}$", color='#bae6fd', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.275, 0.20, "$s$: Hệ số co giãn nhiệt độ học được", color='#7dd3fc', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: Practical value
    ax.add_patch(patches.Rectangle((0.53, 0.15), 0.42, 0.65, fc='#0c4a6e', ec='#38bdf8', lw=1.5, transform=ax.transAxes))
    ax.text(0.74, 0.68, "GIÁ TRỊ THỰC TIỄN (2024-2026)", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.48, "• Ngăn chặn Softmax bão hòa (Entropy Collapse)\n• Bảo vệ huấn luyện FP8 / FP16 không bị tràn số\n• Bắt buộc cho Long-Context 128k - 1M token", color='#e0f2fe', fontsize=8, ha='center', transform=ax.transAxes)
    ax.text(0.74, 0.22, "Áp dụng: Qwen 2.5, OLMo 2, Kimi K2, ViT-22B", color='#34d399', fontsize=8, ha='center', fontweight='bold', transform=ax.transAxes)

    save_and_close(fig, "13_qknorm.png")

# ==========================================
# 14. ADALN-ZERO (DiT / SORA)
# ==========================================
def render_adaln_zero():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "ADALN-ZERO TRONG DIFFUSION TRANSFORMERS (DiT / SORA / FLUX.1)", color='#10b981', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.90, 0.65, fc='#1e293b', ec='#10b981', lw=2, transform=ax.transAxes))
    ax.text(0.5, 0.68, r"$\mathbf{y} = \alpha_c \odot \mathcal{F}\left( \text{LN}(\mathbf{x}) \odot (1 + \gamma_c) + \beta_c \right)$", color='#34d399', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.48, r"$(\gamma_c, \beta_c, \alpha_c) = \text{MLP}(\mathbf{c}) \quad (\mathbf{c}: \text{Timestep } t + \text{Conditioning Prompt})$", color='#ffffff', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, r"Khởi tạo $\alpha_c(0) = \mathbf{0} \to$ Khối DiT bắt đầu như một Identity Mapping thuần túy $\to$ Huấn luyện mượt mà.", color='#a7f3d0', fontsize=8.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "14_adaln_zero.png")

# ==========================================
# 15. LAYERSCALE & DINOV2 REGISTERS
# ==========================================
def render_layerscale_dinov2():
    fig, ax = setup_ax((9.2, 3.0))
    ax.text(0.5, 0.88, "LAYERSCALE & 4 REGISTERS TRONG VISION TRANSFORMERS (DINOv2)", color='#f59e0b', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.90, 0.65, fc='#1e293b', ec='#f59e0b', lw=2, transform=ax.transAxes))
    ax.text(0.5, 0.68, r"$\mathbf{x}' = \mathbf{x} + \text{diag}(\lambda_l) \odot \text{MHSA}(\text{LN}(\mathbf{x})), \quad \lambda_l \sim 10^{-5}$", color='#fbbf24', fontsize=11.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.46, "Kết hợp 4 Register Tokens [R_1, R_2, R_3, R_4] để gom nhiễu nền và ánh sáng", color='#ffffff', fontsize=9.5, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, r"Loại bỏ gai $L_2\text{-norm} > 140$ ở background patch $\to$ Giữ phân phối phẳng đồng nhất $\approx 26.8$.", color='#fde68a', fontsize=8.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "15_layerscale_dinov2.png")

# ==========================================
# 16. DYT & DERF (POINT-WISE SOTA)
# ==========================================
def render_dyt_derf():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "BIÊN GIỚI NORMALIZATION-FREE (2025–2026): DYNAMIC TANH (DyT) & DERF", color='#ec4899', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Left: DyT
    ax.add_patch(patches.Rectangle((0.05, 0.15), 0.43, 0.65, fc='#1e293b', ec='#f59e0b', lw=1.5, transform=ax.transAxes))
    ax.text(0.265, 0.68, "DYNAMIC TANH (DyT - CVPR 2025)", color='#fbbf24', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.265, 0.50, r"$\text{DyT}(\mathbf{x}) = \gamma \odot \tanh(\alpha \odot \mathbf{x}) + \beta$", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.265, 0.28, "Thay thế thống kê toàn cục bằng hàm\nphi tuyến điểm-điểm (Point-wise).\nZero cross-thread synchronization.", color='#fde68a', fontsize=7.5, ha='center', transform=ax.transAxes)

    # Right: Derf
    ax.add_patch(patches.Rectangle((0.52, 0.15), 0.43, 0.65, fc='#1e293b', ec='#ec4899', lw=1.5, transform=ax.transAxes))
    ax.text(0.735, 0.68, "DERF (GAUSSIAN ERROR - 2026)", color='#f472b6', fontsize=9, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.735, 0.50, r"$\text{Derf}(\mathbf{x}) = \text{erf}(\alpha \odot \mathbf{x} + \mathbf{s})$", color='#ffffff', fontsize=9.5, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.735, 0.28, "Sử dụng hàm sai số Gauss rescaled,\nkhớp hoàn hảo phân phối chuẩn tắc.\nTăng 20-30% tốc độ suy luận NPU/GPU.", color='#fbcfe8', fontsize=7.5, ha='center', transform=ax.transAxes)

    save_and_close(fig, "16_dyt_derf.png")

# ==========================================
# 17. HARDWARE MEMORY BANDWIDTH
# ==========================================
def render_hardware_bandwidth():
    fig, ax = setup_ax((9.2, 3.2))
    ax.text(0.5, 0.88, "PHÂN TÍCH HIỆU NĂNG PHẦN CỨNG & BĂNG THÔNG BỘ NHỚ (GPU MEMORY BANDWIDTH)", color='#38bdf8', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # 3 bars comparison
    categories = ['LayerNorm (2 Passes)', 'RMSNorm (1 Pass)', 'DyT/Derf (0 Pass)']
    colors = ['#ef4444', '#a855f7', '#10b981']
    speedups = ['1.00x (Baseline)', '1.35x - 1.50x', '1.50x - 1.75x']
    notes = ['2 lần đọc/ghi DRAM', '1 lần đọc/ghi DRAM', 'Fused hoàn toàn vào ALU']
    
    for i, (cat, col, spd, note) in enumerate(zip(categories, colors, speedups, notes)):
        x_pos = 0.05 + i * 0.31
        ax.add_patch(patches.Rectangle((x_pos, 0.15), 0.28, 0.65, fc='#1e293b', ec=col, lw=1.5, transform=ax.transAxes))
        ax.text(x_pos + 0.14, 0.68, cat, color=col, fontsize=8.5, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(x_pos + 0.14, 0.48, spd, color='#ffffff', fontsize=11, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(x_pos + 0.14, 0.26, note, color='#cbd5e1', fontsize=8, ha='center', transform=ax.transAxes)

    save_and_close(fig, "17_hardware_bandwidth.png")

def main():
    print("🚀 Đang khởi tạo bộ 17 sơ đồ trực quan từng khối cho LayerNorm Research...")
    render_bn_vs_ln()
    render_step1_mean()
    render_step2_var()
    render_step3_standardize()
    render_step4_affine()
    render_invariance_laws()
    render_jacobian_geometry()
    render_loss_landscape()
    render_post_vs_pre_ln()
    render_deepnorm()
    render_parallel_blocks()
    render_rmsnorm()
    render_qknorm()
    render_adaln_zero()
    render_layerscale_dinov2()
    render_dyt_derf()
    render_hardware_bandwidth()
    print("✅ Hoàn tất render toàn bộ 17 hình minh họa từng khối!")

if __name__ == "__main__":
    main()
