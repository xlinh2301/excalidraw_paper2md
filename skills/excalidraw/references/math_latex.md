# Mathematical Equations & LaTeX Rendering in Excalidraw

This guide defines how to author, embed, and render mathematical equations and Deep Learning formulations in Excalidraw compatible with:
1. **ExcaliMath VS Code Extension** (`DynoW.excalimath-editor` / `tamerUAE/excalimath`)
2. **Obsidian-Excalidraw Plugin** (`zsviczian/obsidian-excalidraw-plugin`)
3. **Excalidraw CLI** (`@excalidraw/cli`) and web canvas

---

## 1. Core Architecture: Vector SVG Elements with Metadata

Excalidraw does not render raw markdown math (`$$...$$`) inside text elements natively. To achieve pixel-perfect mathematical typography and full click-to-edit interactivity:

1. **LaTeX formulas are rendered as Vector SVGs** and stored in the root `files` hash table.
2. **Canvas element is an `image` element** pointing to `fileId`.
3. **Metadata is attached** via `customData.excalimath` for round-trip editing in VS Code.

### Element Structure:
```json
{
  "id": "eq-erm-loss",
  "type": "image",
  "x": 100,
  "y": 200,
  "width": 240,
  "height": 45,
  "angle": 0,
  "strokeColor": "transparent",
  "backgroundColor": "transparent",
  "fillStyle": "solid",
  "strokeWidth": 1,
  "strokeStyle": "solid",
  "roughness": 0,
  "opacity": 100,
  "groupIds": [],
  "frameId": null,
  "roundness": null,
  "seed": 1001,
  "version": 1,
  "versionNonce": 1001,
  "isDeleted": false,
  "boundElements": null,
  "updated": 1,
  "link": null,
  "locked": false,
  "status": "saved",
  "fileId": "file_hash_id_123",
  "scale": [1, 1],
  "customData": {
    "excalimath": {
      "type": "latex",
      "latex": "\\min_{\\boldsymbol{\\theta}} \\mathcal{L}(\\boldsymbol{\\theta})"
    }
  }
}
```

### File Registry Entry (`files` object in root):
```json
{
  "files": {
    "file_hash_id_123": {
      "mimeType": "image/svg+xml",
      "id": "file_hash_id_123",
      "dataURL": "data:image/svg+xml;utf8,<svg ...>...</svg>",
      "created": 1728448000000,
      "lastRetrieved": 1728448000000
    }
  }
}
```

---

## 2. LaTeX Generation via Matplotlib Mathtext / KaTeX

When generating SVGs programmatically in Python:
- Use `matplotlib.figure.Figure` with `bbox_inches='tight'`, `transparent=True`.
- Render with `dpi=200` to `300` for ultra-crisp display.
- Colorize equations using semantic palette:
  - Deep Navy / Primary: `#1971c2`
  - Emerald Green / Accent: `#099268`
  - Amber Orange / Attention: `#d9480f`
  - Dark Charcoal / Body: `#212529`

```python
import io
import hashlib
import matplotlib.pyplot as plt

def render_latex_to_svg(latex_str: str, color="#1971c2", fontsize=14):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0.0)
    
    # Render with Mathtext engine
    text_obj = fig.text(
        0, 0, f"${latex_str}$",
        fontsize=fontsize,
        color=color,
        usetex=False
    )
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close(fig)
    
    svg_content = buf.getvalue().decode('utf-8')
    file_id = hashlib.sha1(svg_content.encode('utf-8')).hexdigest()[:16]
    return file_id, svg_content
```

---

## 3. Supported LaTeX Notation & Rules
- Always use `\mathbf{...}` for vectors and `\boldsymbol{\theta}` for Greek parameter vectors.
- Use `\leq` and `\geq` instead of `\le` and `\ge` for full Mathtext engine compatibility.
- Ensure proper sizing on canvas:
  - Inline / small definitions: `height: 24 - 32px`
  - Main display equations: `height: 38 - 50px`
  - Complex fraction / matrix equations: `height: 55 - 75px`
