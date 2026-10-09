# Deep Learning & Neural Architecture Visualization Guide

This guide establishes the production standards for creating publication-grade Deep Learning and AI Architecture diagrams in Excalidraw, Obsidian-Excalidraw, and vector SVG/PNG formats.

---

## 1. Core Architectural Paradigms

### A. 3D Isometric Tensor & Feature Map Pipeline
- **True 3-Tone Isometric Shading**:
  - **Top Face**: Bright highlight tint (e.g., `#38bdf8`, `#34d399`, `#fbbf24`).
  - **Front Face**: Medium neutral tone (e.g., `#0284c7`, `#059669`, `#d97706`).
  - **Side Face (Right)**: Darker shaded tone (e.g., `#0369a1`, `#047857`, `#b45309`).
  - **Isometric Projection Angles**: $dx = \text{depth} \times 0.70$, $dy = \text{depth} \times 0.40$.
- **Surface Gridlines**: Add faint gridlines on the front face of each tensor slice to illustrate spatial resolution (e.g., $16\times 16$, $8\times 8$, $4\times 4$).
- **Receptive Field Frustums**: Pyramidal polygons connecting source patch bounding boxes to target feature tensors (`fillColor` with low alpha $\sim 0.18$, dashed boundary stroke).
- **3D Flatten Column**: Vertical 3D prism with segmented cells representing feature serialization.
- **Dense Synaptic Graph**: Multi-layer neural network with glowing radial halos, synaptic connection lines, and wide HUD progress-bar badges for calibrated softmax outputs.

---

## 2. Vision Transformer (ViT) Macro & Micro Architecture

### A. Shared Central Vertical Axis Alignment
To avoid visually crooked diagrams, all stacked components in a column MUST share the exact same central vertical coordinate $X_c$:
$$\text{Card } X = X_c - \frac{W_{\text{card}}}{2}$$
$$\text{Arrow Start} = (X_c, Y_{\text{src}} + H_{\text{src}}), \quad \text{Arrow End} = (X_c, Y_{\text{tgt}})$$

### B. Symmetrical Branching (Q, K, V)
Position parallel branches symmetrically around $X_c$:
- **Key ($W_K$)**: Centered at $X_c$.
- **Query ($W_Q$)**: Centered at $X_c - \Delta$.
- **Value ($W_V$)**: Centered at $X_c + \Delta$.

### C. Snap-to-Block Arrow Binding (`startBinding` & `endBinding`)
Every connecting arrow MUST be explicitly bound to its source and target elements:
```json
{
  "id": "arrow-1",
  "type": "arrow",
  "startBinding": { "elementId": "card-src", "focus": 0.0, "gap": 1 },
  "endBinding": { "elementId": "card-tgt", "focus": 0.0, "gap": 1 }
}
```
And on the source/target cards:
```json
{
  "id": "card-src",
  "boundElements": [{ "id": "arrow-1", "type": "arrow" }]
}
```

### D. Residual Skip Connections (Right-Angled Elbow Arrows)
- Route skip bypasses through a dedicated channel at $X_{\text{bypass}} = X_{\text{card\_max}} + \delta$.
- Use `elbowed: true` and 4-point paths:
  $$[X_c, Y_{\text{start}}] \to [X_{\text{bypass}}, Y_{\text{start}}] \to [X_{\text{bypass}}, Y_{\text{add}}] \to [X_c + R_{\text{add}}, Y_{\text{add}}]$$
- Place a dedicated label pill (e.g. `Residual Skip 1`) centered along the horizontal routing segment.

### E. Zoom-in Frustums (Dashed Trapezoids)
When detailing a micro block (e.g., Layer $l \in [1..12]$) expanded from a macro stack:
- Origin: Left edge of the specific block card in the macro stack.
- Destination: Top-left and bottom-left corners of the expanded micro block container.
- Stroke: `dashed`, width `2.0`, matching the module color palette.

---

## 3. Deep-Dive Component Explanations (Grid Layout)

Organize architectural component breakdowns into a $2\times 3$ or $3\times 2$ grid with:
1. **Pre-LayerNorm (Pre-LN)**: Formulations for $\mathrm{LN}(\mathbf{x})$, identity gradient flow $\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \mathbf{I} + \dots$, and training stability.
2. **MHSA + 4 Register Tokens**: Sequence matrix $\mathbf{Z}_0 \in \mathbb{R}^{261\times 768}$, attention entropy $\mathcal{H}(\mathbf{A}_i)$, and absorption of attention sinks (Darcet et al., ICLR 2024).
3. **LayerScale Stabilization**: Diagonal learnable scaling $\mathrm{diag}(\boldsymbol{\lambda}_l) \odot \mathbf{x}$ initialized at $\mathcal{O}(10^{-5})$ (Touvron et al., ICCV 2021).
4. **Residual Skip Dynamics**: Additive shortcut $\mathbf{x} \oplus \mathcal{T}(\mathbf{x})$ and isotropic Hessian condition number $\kappa(\mathbf{H}) \approx 1.0$.
5. **SwiGLU Gated Feed-Forward**: Multiplicative gate $(\mathrm{Swish}(\mathbf{x}W_1) \odot \mathbf{x}W_3)W_2$ with $d_{\text{ffn}} = \frac{8}{3}D = 2048$.
6. **Multi-Token Pooling & Classification Head**: Tri-stream concatenation $[\mathbf{z}_{\text{cls}} \,\|\, \frac{1}{4}\sum \mathbf{z}_{\text{reg}} \,\|\, \frac{1}{N}\sum \mathbf{z}_{\text{patch}}] \in \mathbb{R}^{2304}$ and ISO/IEC 30107-3 compliance.

---

## 4. Typography & Color Standards

- **Primary Sans-serif**: `fontFamily: 2` (Clean Helvetica/Arial) for all titles, badges, and card text.
- **Code & Benchmarks**: `fontFamily: 3` (Cascadia/Monospace) for data tables and tensor shapes.
- **Math Formulas**: True LaTeX vector SVGs generated at $\text{DPI} \ge 220$ with `customData.excalimath.latex` metadata.
- **Theme**: Dark Hero (`#090d16` canvas, `#0f172a` cards, vibrant neon accent strokes `#38bdf8`, `#34d399`, `#c084fc`, `#fbbf24`, `#f43f5e`).
