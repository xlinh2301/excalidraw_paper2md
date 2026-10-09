#!/usr/bin/env python3
"""
Batch Convert Foundational & SOTA Normalization Papers to Markdown with Original Figures
========================================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from paper2md import convert_paper_universal

PAPERS_TO_INGEST = [
    {"id": "1607.06450", "name": "01_layernorm_ba_hinton_2016", "title": "Layer Normalization (Ba, Kiros, Hinton)"},
    {"id": "1910.07467", "name": "02_rmsnorm_zhang_2019", "title": "Root Mean Square Layer Normalization (Zhang & Sennrich)"},
    {"id": "2203.00555", "name": "03_deepnorm_wang_2022", "title": "DeepNet: Scaling Transformers to 1,000 Layers (Microsoft)"},
    {"id": "2309.16588", "name": "04_dinov2_registers_darcet_2023", "title": "Vision Transformers Need Registers (Darcet et al.)"},
    {"id": "2212.09748", "name": "05_dit_adaln_peebles_2022", "title": "Scalable Diffusion Models with Transformers (DiT / AdaLN)"},
    {"id": "1803.08494", "name": "06_groupnorm_wu_he_2018", "title": "Group Normalization (Wu & Kaiming He)"},
    {"id": "1802.05957", "name": "07_spectralnorm_miyato_2018", "title": "Spectral Normalization for GANs (Miyato et al.)"},
    {"id": "1502.03167", "name": "08_batchnorm_ioffe_szegedy_2015", "title": "Batch Normalization (Ioffe & Szegedy)"},
    {"id": "1602.07868", "name": "09_weightnorm_salimans_2016", "title": "Weight Normalization (Salimans & Kingma)"}
]

def main():
    base_out = PROJECT_ROOT / "docs" / "research" / "converted_papers"
    base_out.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Bắt đầu tải và chuyển đổi {len(PAPERS_TO_INGEST)} papers sang Markdown + Trích xuất ảnh gốc...\n")
    
    success_count = 0
    for p in PAPERS_TO_INGEST:
        out_dir = base_out / p["name"]
        print(f"================================================================================")
        print(f"[*] Processing: {p['title']} ({p['id']})")
        print(f"    Target Dir: {out_dir}")
        print(f"================================================================================")
        
        res = convert_paper_universal(p["id"], out_dir)
        if res and res.exists():
            success_count += 1
            print(f"✅ Success: {p['name']}\n")
        else:
            print(f"❌ Failed: {p['name']}\n")
            
    print(f"\n🎉 HOÀN TẤT BATCH CONVERSION: {success_count}/{len(PAPERS_TO_INGEST)} papers converted successfully!")

if __name__ == "__main__":
    main()
