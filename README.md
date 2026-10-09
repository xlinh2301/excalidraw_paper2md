# Excalidraw & Paper2MD Skills Suite (`excalidraw_paper2md`)

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI Agent Ready](https://img.shields.io/badge/AI%20Agents-Antigravity%20%7C%20Claude%20%7C%20Codex-purple.svg)](https://github.com/xlinh2301/excalidraw_paper2md)
[![Markdown & LaTeX](https://img.shields.io/badge/LaTeX-Standard%20%24...%24-orange.svg)](https://github.com/xlinh2301/excalidraw_paper2md)

**A dual-engine toolkit combining Universal Paper-to-Markdown Ingestion with 100% Figure Extraction and Excalidraw Deep Learning Architecture Diagramming.**

[Features](#-key-features) • [Installation](#-installation) • [Paper2MD Engine](#-paper2md-universal-paper-converter) • [Excalidraw Diagramming](#-excalidraw-architecture-engine) • [Agent Integration](#-agent-skills-integration) • [License](#-license)

</div>

---

## 🌟 Overview

`excalidraw_paper2md` is a modular toolkit designed for AI researchers, autonomous coding agents (Antigravity, Claude Code, OpenAI Codex), and technical writers. It bridges two critical gaps in technical workflows:

1. **`paper2md`**: Converts any scientific paper (arXiv ID, arXiv URL, remote PDF URL, or local PDF) into clean, GitHub-flavored Markdown with **standard LaTeX math formulas (`$...$`, `$$...$$`)** and **extracts 100% of authentic author figures, graphs, and vectors** (auto-converting PDF/EPS vectors to high-resolution PNGs).
2. **`excalidraw`**: Programmatically generates beautiful, hand-drawn architecture diagrams, state machines, and tensor geometry visuals in `.excalidraw` (JSON), `.png`, and `.svg` formats, with native Obsidian wiki-links integration.

---

## 🚀 Key Features

### 📄 1. `paper2md` (Universal Paper-to-Markdown Engine)
- **Multi-Source Ingestion**: Accepts raw arXiv IDs (`2304.07193`), arXiv abstract/PDF URLs, public web PDF URLs, or local `.pdf` files.
- **Dual-Engine Architecture**:
  - **Primary (arXiv LaTeX Source)**: Downloads and parses official LaTeX e-print sources (`.tex`, `.bib`, `.sty`), extracting the author's original high-res figures and converting `.pdf`/`.eps` vector graphics to crisp PNGs.
  - **Fallback (PyMuPDF Layout Engine)**: For non-arXiv or scanned PDFs, uses advanced PyMuPDF layout analysis to extract text blocks, headings, formulas, and embedded raster images.
- **LaTeX Math Formula Preservation**: Automatically standardizes LaTeX equations into clean inline (`$...$`) and display (`$$...$$`) math blocks compatible with GitHub Flavored Markdown and Obsidian.
- **Batch Processing**: Convert entire lists or directories of papers in parallel with `batch_convert_papers.py`.

### 🎨 2. `excalidraw` (Architectural Diagramming Engine)
- **Deep Learning Block Generation**: Scripts to generate Transformer blocks, Multi-Head Self-Attention (MHSA), Pre-LayerNorm / Post-LayerNorm, LayerScale, Residual streams, and SwiGLU FFNs.
- **Tensor Geometry Visuals**: Visualize 3D/4D tensor slicing across batch, channel, and spatial dimensions (BatchNorm vs LayerNorm vs GroupNorm vs InstanceNorm).
- **Multi-Format Export**: Generates `.excalidraw` editable scene files, high-resolution `.png`, and scalable `.svg`.
- **Obsidian Ecosystem**: Automatic frontmatter generation and Markdown embedding.

---

## 📦 Installation

```bash
# Clone repository
git clone git@github.com:xlinh2301/excalidraw_paper2md.git
cd excalidraw_paper2md

# Install dependencies
pip install -r requirements.txt
```

### Requirements:
- Python 3.8+
- `PyMuPDF` (`fitz` >= 1.23.0)
- `Pillow` (>= 9.5.0)
- `requests` (>= 2.31.0)
- `tqdm` (>= 4.66.0)
- `cairosvg` (>= 2.7.0, optional for SVG rendering)

---

## 📖 `paper2md` Universal Paper Converter

### CLI Usage

```bash
# Convert by arXiv ID
python3 paper2md.py 2304.07193 -o output/dinov2/

# Convert by arXiv URL
python3 paper2md.py https://arxiv.org/abs/2104.14294 -o output/dino/

# Convert by local PDF file
python3 paper2md.py /path/to/paper.pdf -o output/local_paper/
```

### Python API

```python
from paper2md import convert_paper_to_markdown

# Convert any paper
output_path = convert_paper_to_markdown(
    input_source="2003.04092",
    output_dir="./converted_papers/cdcn_2020"
)
print(f"Markdown saved to: {output_path}")
```

### Output Structure
```
output/dinov2/
├── paper.md                # Clean GitHub-flavored Markdown with math
└── figures/                # 100% original author figures
    ├── new-figure-1.jpg
    ├── LaViDa_datapipeline_figure.png
    └── pullfigure_5.png
```

---

## 🎨 `excalidraw` Architecture Engine

### Generate Deep Learning Architectures
```bash
# Generate Transformer / LayerScale / MHSA blocks
python3 skills/excalidraw/scripts/generate_deep_learning_architecture.py

# Generate LayerNorm block breakdown diagrams
python3 skills/excalidraw/scripts/generate_layernorm_block_diagrams.py

# Generate 3D Tensor Slicing & Loss Contour geometry diagrams
python3 skills/excalidraw/scripts/generate_normalization_geometry_diagrams.py
```

---

## 🤖 Agent Skills Integration

This repository conforms to the standard Agent Skill packaging specification for:
- **Google Antigravity (`.agents/skills/`)**
- **Claude Code (`.claude/skills/`)**
- **OpenAI Codex & Custom AI Assistants**

### Skill Manifests Included:
- [`skills/paper2md/SKILL.md`](skills/paper2md/SKILL.md): Ingestion guidelines, LaTeX handling rules, figure extraction protocols.
- [`skills/excalidraw/SKILL.md`](skills/excalidraw/SKILL.md): Visual hierarchy, color palettes, bounding box mathematics, arrow bindings.

---

## 📂 Repository Structure

```
excalidraw_paper2md/
├── .gitignore
├── README.md
├── requirements.txt
├── paper2md.py                     # Root CLI entry point
├── scripts/
│   └── paper2md.py
└── skills/
    ├── paper2md/
    │   ├── SKILL.md                # Agent Skill definition
    │   ├── references/             # Ingestion & conversion references
    │   └── scripts/
    │       ├── paper2md.py
    │       └── batch_convert_papers.py
    └── excalidraw/
        ├── SKILL.md                # Agent Skill definition
        ├── README.md
        ├── references/             # Colors, arrows, math, JSON schemas
        └── scripts/                # Architecture generation scripts
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Developed with ❤️ by <a href="https://github.com/xlinh2301">xlinh2301</a>
</div>
