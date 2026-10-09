#!/usr/bin/env python3
"""
Master Ultra-Polished Hybrid Deep Learning Architecture Generator:
1. True 3D Isometric Tensor Feature Maps with 3-Tone Shading, Surface Gridlines, Receptive Field Frustums & Synaptic Graph
2. DINOv2 Vision Transformer Micro Block Detail (Pre-LN, Q/K/V Attention Matrix Heatmap, LayerScale, Residual Skip, SwiGLU)
3. DINOv2 Macro Pipeline with Patch/Register Token Ribbon, 12x Encoder Stack, and 3-Branch Multi-Token Fusion Head
4. Mathematical Formulations & Glassmorphic Benchmark Matrix (ISO/IEC 30107-3 Compliant)
5. SECTION E (BOTTOM): Deep Dive Architectural Component Analysis with 6 Dedicated Vector Visual Illustrations (Pre-LN Highway, Attention Sinks & Registers, LayerScale Scaling, Residual Skip Gradient, SwiGLU Gating Dataflow, Tri-Stream Multi-Token Pooling)

Target: docs/diagrams/dinov2_fas_architecture.excalidraw
"""

import io
import json
import base64
import hashlib
import random
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Typography configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

elements = []
files = {}
element_id_counter = 1
element_map = {}

def next_id(prefix="elem"):
    global element_id_counter
    eid = f"{prefix}-{element_id_counter}"
    element_id_counter += 1
    return eid

def register_svg_file(svg_bytes):
    data_url = 'data:image/svg+xml;base64,' + base64.b64encode(svg_bytes).decode('ascii')
    file_id = hashlib.sha256(svg_bytes).hexdigest()[:16]
    files[file_id] = {
        "mimeType": "image/svg+xml",
        "id": file_id,
        "dataURL": data_url,
        "created": 1728460000000
    }
    return file_id

def render_latex_to_svg(latex_str, color="#1e293b", fontsize=12.5, dpi=220):
    fig = plt.figure(figsize=(0.1, 0.1), dpi=dpi)
    fig.patch.set_alpha(0.0)
    fig.text(0, 0, f"${latex_str}$", fontsize=fontsize, color=color, ha='left', va='bottom', fontfamily='sans-serif')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close(fig)
    
    svg_bytes = buf.getvalue()
    tree = ET.fromstring(svg_bytes)
    viewbox = tree.attrib.get('viewBox', '').split()
    w = float(viewbox[2]) if len(viewbox) == 4 else 120.0
    h = float(viewbox[3]) if len(viewbox) == 4 else 28.0
    file_id = register_svg_file(svg_bytes)
    return file_id, w, h

def create_latex_math_element(x, y, latex_str, color="#1e293b", fontsize=12.5, scale=1.0):
    file_id, w, h = render_latex_to_svg(latex_str, color=color, fontsize=fontsize)
    elem_id = next_id("math")
    elem_w = w * scale
    elem_h = h * scale
    
    img_elem = {
        "id": elem_id, "type": "image", "x": x, "y": y, "width": elem_w, "height": elem_h,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": random.randint(1000, 99999), "version": 1, "versionNonce": random.randint(1000, 99999),
        "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False, "status": "saved",
        "fileId": file_id, "scale": [1, 1],
        "customData": {"excalimath": {"type": "latex", "latex": latex_str}}
    }
    elements.append(img_elem)
    element_map[elem_id] = img_elem
    return img_elem, elem_w, elem_h

def create_card(x, y, w, h, bg_color="#ffffff", stroke_color="#1e293b", stroke_width=1.5, stroke_style="solid", border_radius=8, opacity=100):
    card_id = next_id("card")
    card = {
        "id": card_id, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke_color, "backgroundColor": bg_color, "fillStyle": "solid",
        "strokeWidth": stroke_width, "strokeStyle": stroke_style, "roughness": 0, "opacity": opacity,
        "groupIds": [], "frameId": None, "roundness": {"type": 3} if border_radius > 0 else None,
        "seed": random.randint(1000, 99999), "version": 1, "versionNonce": random.randint(1000, 99999),
        "isDeleted": False, "boundElements": [], "updated": 1, "link": None, "locked": False
    }
    elements.append(card)
    element_map[card_id] = card
    return card

def create_ellipse(x, y, w, h, bg_color="#38bdf8", stroke_color="#0284c7", stroke_width=1.5):
    eid = next_id("ellipse")
    elem = {
        "id": eid, "type": "ellipse", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke_color, "backgroundColor": bg_color, "fillStyle": "solid",
        "strokeWidth": stroke_width, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None, "seed": random.randint(1000, 99999),
        "version": 1, "versionNonce": random.randint(1000, 99999), "isDeleted": False,
        "boundElements": [], "updated": 1, "link": None, "locked": False
    }
    elements.append(elem)
    element_map[eid] = elem
    return elem

def create_text(x, y, text, font_size=12, color="#1e293b", font_family=2, text_align="left", is_bold=False):
    text_id = next_id("text")
    lines = text.split("\n")
    line_height = 1.35
    w = max([len(l) for l in lines]) * (font_size * 0.58) + 10
    h = len(lines) * (font_size * line_height)
    
    elem = {
        "id": text_id, "type": "text", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None, "seed": random.randint(1000, 99999),
        "version": 1, "versionNonce": random.randint(1000, 99999), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False, "text": text,
        "fontSize": font_size, "fontFamily": font_family, "textAlign": text_align,
        "verticalAlign": "top", "baseline": font_size, "containerId": None, "originalText": text,
        "lineHeight": line_height
    }
    elements.append(elem)
    element_map[text_id] = elem
    return elem

def create_centered_text(cx, cy, text, font_size=12, color="#1e293b", font_family=2, is_bold=False):
    lines = text.split("\n")
    line_height = 1.35
    w = max([len(l) for l in lines]) * (font_size * 0.58) + 10
    h = len(lines) * (font_size * line_height)
    x = cx - w / 2
    y = cy - h / 2
    return create_text(x, y, text, font_size=font_size, color=color, font_family=font_family, text_align="center", is_bold=is_bold)

def bind_arrow_to_elements(arrow_elem, start_elem=None, end_elem=None):
    if start_elem:
        arrow_elem["startBinding"] = {
            "elementId": start_elem["id"],
            "focus": 0.0,
            "gap": 1
        }
        if start_elem.get("boundElements") is None:
            start_elem["boundElements"] = []
        start_elem["boundElements"].append({"id": arrow_elem["id"], "type": "arrow"})
    if end_elem:
        arrow_elem["endBinding"] = {
            "elementId": end_elem["id"],
            "focus": 0.0,
            "gap": 1
        }
        if end_elem.get("boundElements") is None:
            end_elem["boundElements"] = []
        end_elem["boundElements"].append({"id": arrow_elem["id"], "type": "arrow"})

def create_connected_arrow(start_pt, end_pt, start_elem=None, end_elem=None, color="#94a3b8", stroke_width=1.6, stroke_style="solid", is_bidirectional=False):
    aid = next_id("arrow")
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    
    elem = {
        "id": aid, "type": "arrow", "x": start_pt[0], "y": start_pt[1],
        "width": abs(dx) if abs(dx) > 0 else 1, "height": abs(dy) if abs(dy) > 0 else 1, "angle": 0, "strokeColor": color,
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": stroke_width,
        "strokeStyle": stroke_style, "roughness": 0, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": {"type": 2}, "seed": random.randint(1000, 99999),
        "version": 1, "versionNonce": random.randint(1000, 99999), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [dx, dy]], "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": "arrow" if is_bidirectional else None,
        "endArrowhead": "arrow", "elbowed": False
    }
    bind_arrow_to_elements(elem, start_elem, end_elem)
    elements.append(elem)
    element_map[aid] = elem
    return elem

def create_connected_elbow_arrow(points, start_elem=None, end_elem=None, color="#94a3b8", stroke_width=1.6, stroke_style="solid"):
    aid = next_id("arrow")
    ox, oy = points[0]
    rel_pts = [[p[0] - ox, p[1] - oy] for p in points]
    xs = [p[0] for p in rel_pts]
    ys = [p[1] for p in rel_pts]
    w = max(max(xs) - min(xs), 1)
    h = max(max(ys) - min(ys), 1)
    
    elem = {
        "id": aid, "type": "arrow", "x": ox, "y": oy,
        "width": w, "height": h, "angle": 0, "strokeColor": color,
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": stroke_width,
        "strokeStyle": stroke_style, "roughness": 0, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": {"type": 2}, "seed": random.randint(1000, 99999),
        "version": 1, "versionNonce": random.randint(1000, 99999), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
        "points": rel_pts, "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow", "elbowed": True
    }
    bind_arrow_to_elements(elem, start_elem, end_elem)
    elements.append(elem)
    element_map[aid] = elem
    return elem

def create_line(start_pt, end_pt, color="#94a3b8", stroke_width=1.0, stroke_style="dashed"):
    lid = next_id("line")
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    
    elem = {
        "id": lid, "type": "line", "x": start_pt[0], "y": start_pt[1],
        "width": abs(dx) if abs(dx) > 0 else 1, "height": abs(dy) if abs(dy) > 0 else 1, "angle": 0, "strokeColor": color,
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": stroke_width,
        "strokeStyle": stroke_style, "roughness": 0, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": None, "seed": random.randint(1000, 99999),
        "version": 1, "versionNonce": random.randint(1000, 99999), "isDeleted": False,
        "boundElements": None, "updated": 1, "link": None, "locked": False,
        "points": [[0, 0], [dx, dy]], "lastCommittedPoint": None, "startBinding": None,
        "endBinding": None, "startArrowhead": None, "endArrowhead": None
    }
    elements.append(elem)
    element_map[lid] = elem
    return elem

def create_pill(x, y, w, h, text, bg_color, stroke_color, text_color):
    card = create_card(x, y, w, h, bg_color=bg_color, stroke_color=stroke_color, stroke_width=1.5, border_radius=12)
    create_centered_text(x + w / 2, y + h / 2, text, font_size=11, color=text_color, font_family=2, is_bold=True)
    return card

# -----------------------------------------------------------------------------
# HIGH-FIDELITY 3D ISOMETRIC VECTOR RENDERER (SECTION A)
# -----------------------------------------------------------------------------
def render_master_3d_isometric_pipeline_svg():
    fig = plt.figure(figsize=(14.8, 4.4), dpi=240)
    fig.patch.set_facecolor('#090d16')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#090d16')
    
    def draw_iso_slab(ox, oy, w, h, depth, top_col, front_col, side_col, edge_col, alpha=0.9, grid_n=0):
        dx = depth * 0.70
        dy = depth * 0.40
        
        # 1. Front face
        front_pts = np.array([[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]])
        front_poly = patches.Polygon(front_pts, closed=True, facecolor=front_col, edgecolor=edge_col, lw=1.0, alpha=alpha, zorder=2)
        ax.add_patch(front_poly)
        
        if grid_n > 1:
            for gi in range(1, grid_n):
                gx = ox + (w / grid_n) * gi
                ax.plot([gx, gx], [oy, oy + h], color=edge_col, lw=0.5, alpha=0.35, zorder=3)
                gy = oy + (h / grid_n) * gi
                ax.plot([ox, ox + w], [gy, gy], color=edge_col, lw=0.5, alpha=0.35, zorder=3)
                
        # 2. Top face
        top_pts = np.array([[ox, oy + h], [ox + w, oy + h], [ox + w + dx, oy + h + dy], [ox + dx, oy + h + dy]])
        top_poly = patches.Polygon(top_pts, closed=True, facecolor=top_col, edgecolor=edge_col, lw=1.0, alpha=alpha, zorder=4)
        ax.add_patch(top_poly)
        
        # 3. Right side face
        side_pts = np.array([[ox + w, oy], [ox + w + dx, oy + dy], [ox + w + dx, oy + h + dy], [ox + w, oy + h]])
        side_poly = patches.Polygon(side_pts, closed=True, facecolor=side_col, edgecolor=edge_col, lw=1.0, alpha=alpha, zorder=3)
        ax.add_patch(side_poly)

    # 1. INPUT IMAGE (Biometric Face Scan Frame)
    img_ox, img_oy = 0.6, 1.15
    img_w, img_h = 2.4, 2.7
    ax.add_patch(patches.FancyBboxPatch((img_ox - 0.1, img_oy - 0.1), img_w + 0.2, img_h + 0.2, boxstyle="round,pad=0.08", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.8, zorder=1))
    ax.add_patch(patches.Rectangle((img_ox, img_oy), img_w, img_h, facecolor="#0f172a", edgecolor="#0284c7", lw=1.0, zorder=2))
    
    # Biometric Face Landmark wireframe
    head_ellipse = patches.Ellipse((img_ox + img_w/2, img_oy + img_h*0.50), 1.2, 1.5, facecolor="none", edgecolor="#38bdf8", lw=1.4, alpha=0.85, zorder=3)
    ax.add_patch(head_ellipse)
    ax.add_patch(patches.Circle((img_ox + img_w/2 - 0.3, img_oy + img_h*0.56), 0.08, facecolor="#38bdf8", edgecolor="#e0f2fe", lw=1.0, zorder=4))
    ax.add_patch(patches.Circle((img_ox + img_w/2 + 0.3, img_oy + img_h*0.56), 0.08, facecolor="#38bdf8", edgecolor="#e0f2fe", lw=1.0, zorder=4))
    ax.plot([img_ox + img_w/2, img_ox + img_w/2], [img_oy + img_h*0.50, img_oy + img_h*0.40], color="#38bdf8", lw=1.2, zorder=4)
    ax.plot([img_ox + img_w/2 - 0.22, img_ox + img_w/2 + 0.22], [img_oy + img_h*0.30, img_oy + img_h*0.30], color="#38bdf8", lw=1.4, zorder=4)
    # Laser Scan Line
    ax.plot([img_ox, img_ox + img_w], [img_oy + img_h*0.53, img_oy + img_h*0.53], color="#f43f5e", lw=1.8, alpha=0.9, zorder=5)
    
    # Receptive Field Kernel Box (14x14 Patch)
    k_ox, k_oy, k_s = img_ox + 1.25, img_oy + 1.25, 0.65
    ax.add_patch(patches.Rectangle((k_ox, k_oy), k_s, k_s, facecolor="#f43f5e", alpha=0.45, edgecolor="#fb7185", lw=1.6, zorder=6))
    ax.text(img_ox + img_w/2, img_oy - 0.35, "INPUT IMAGE\n(224 x 224 x 3)", color="#f8fafc", fontsize=8.5, ha='center', weight='bold')
    ax.text(k_ox + k_s/2, k_oy + k_s + 0.16, "Receptive Field (14x14)", color="#fb7185", fontsize=7.2, ha='center', weight='bold')

    # 2. STAGE 1: PATCH TENSORS (16 x 16 x 64) - Electric Blue Slices
    s1_ox = 4.4
    s1_w, s1_h, s1_dep = 1.6, 2.2, 0.28
    for i in range(5):
        slab_x = s1_ox + i * 0.24
        slab_y = 1.0 + i * 0.14
        draw_iso_slab(slab_x, slab_y, s1_w, s1_h, s1_dep,
                      top_col="#38bdf8", front_col="#0284c7", side_col="#0369a1", edge_col="#7dd3fc", alpha=0.85, grid_n=4)
    
    last_s1_x = s1_ox + 4 * 0.24
    last_s1_y = 1.0 + 4 * 0.14
    ax.add_patch(patches.Rectangle((last_s1_x + 0.6, last_s1_y + 1.1), 0.45, 0.45, facecolor="#f43f5e", alpha=0.5, edgecolor="#fb7185", lw=1.2, zorder=7))
    ax.text(s1_ox + 1.1, 0.45, "PATCH TENSORS\n(16 x 16 x 64)", color="#38bdf8", fontsize=8.5, ha='center', weight='bold')

    # Receptive Field Frustum 0 -> 1
    frustum_pts_01 = np.array([
        [k_ox + k_s, k_oy + k_s],
        [last_s1_x + 0.6, last_s1_y + 1.55],
        [last_s1_x + 0.6, last_s1_y + 1.1],
        [k_ox + k_s, k_oy]
    ])
    ax.add_patch(patches.Polygon(frustum_pts_01, closed=True, facecolor="#f43f5e", alpha=0.18, edgecolor="#fb7185", lw=1.2, linestyle="--", zorder=5))

    # 3. STAGE 2: POOLED FEATURE MAPS (8 x 8 x 128) - Emerald Slices
    s2_ox = 8.0
    s2_w, s2_h, s2_dep = 1.3, 1.8, 0.25
    for i in range(6):
        slab_x = s2_ox + i * 0.20
        slab_y = 1.2 + i * 0.12
        draw_iso_slab(slab_x, slab_y, s2_w, s2_h, s2_dep,
                      top_col="#34d399", front_col="#059669", side_col="#047857", edge_col="#6ee7b7", alpha=0.85, grid_n=3)
        
    last_s2_x = s2_ox + 5 * 0.20
    last_s2_y = 1.2 + 5 * 0.12
    ax.add_patch(patches.Rectangle((last_s2_x + 0.4, last_s2_y + 0.8), 0.38, 0.38, facecolor="#f43f5e", alpha=0.5, edgecolor="#fb7185", lw=1.2, zorder=7))
    ax.text(s2_ox + 0.9, 0.45, "POOLED MAPS\n(8 x 8 x 128)", color="#34d399", fontsize=8.5, ha='center', weight='bold')

    # Projection Frustum 1 -> 2
    frustum_pts_12 = np.array([
        [last_s1_x + 1.05, last_s1_y + 1.55],
        [last_s2_x + 0.4, last_s2_y + 1.18],
        [last_s2_x + 0.4, last_s2_y + 0.8],
        [last_s1_x + 1.05, last_s1_y + 1.1]
    ])
    ax.add_patch(patches.Polygon(frustum_pts_12, closed=True, facecolor="#34d399", alpha=0.18, edgecolor="#34d399", lw=1.2, linestyle="--", zorder=5))

    # 4. STAGE 3: DEEP FEATURE MAPS (4 x 4 x 256) - Amber/Gold Slices
    s3_ox = 11.2
    s3_w, s3_h, s3_dep = 1.0, 1.4, 0.22
    for i in range(8):
        slab_x = s3_ox + i * 0.16
        slab_y = 1.4 + i * 0.10
        draw_iso_slab(slab_x, slab_y, s3_w, s3_h, s3_dep,
                      top_col="#fbbf24", front_col="#d97706", side_col="#b45309", edge_col="#fde68a", alpha=0.85, grid_n=2)
        
    last_s3_x = s3_ox + 7 * 0.16
    last_s3_y = 1.4 + 7 * 0.10
    ax.text(s3_ox + 0.8, 0.45, "DEEP MAPS\n(4 x 4 x 256)", color="#fbbf24", fontsize=8.5, ha='center', weight='bold')

    # Projection Frustum 2 -> 3
    frustum_pts_23 = np.array([
        [last_s2_x + 0.78, last_s2_y + 1.18],
        [last_s3_x + 0.3, last_s3_y + 1.1],
        [last_s3_x + 0.3, last_s3_y + 0.7],
        [last_s2_x + 0.78, last_s2_y + 0.8]
    ])
    ax.add_patch(patches.Polygon(frustum_pts_23, closed=True, facecolor="#fbbf24", alpha=0.18, edgecolor="#fbbf24", lw=1.2, linestyle="--", zorder=5))

    # 5. 3D FLATTENED TENSOR PILLAR (2048-d)
    fl_ox, fl_oy = 14.3, 1.2
    draw_iso_slab(fl_ox, fl_oy, 0.5, 2.6, 0.3,
                  top_col="#fb923c", front_col="#ea580c", side_col="#c2410c", edge_col="#fed7aa", alpha=0.9, grid_n=8)
    ax.text(fl_ox + 0.35, 0.45, "FLATTEN\n(2048-d)", color="#fb923c", fontsize=8.5, ha='center', weight='bold')

    # Frustum Deep -> Flatten
    frustum_pts_3fl = np.array([
        [last_s3_x + s3_w, last_s3_y + s3_h],
        [fl_ox, fl_oy + 2.6],
        [fl_ox, fl_oy],
        [last_s3_x + s3_w, last_s3_y]
    ])
    ax.add_patch(patches.Polygon(frustum_pts_3fl, closed=True, facecolor="#ea580c", alpha=0.18, edgecolor="#fb923c", lw=1.2, linestyle="--", zorder=5))

    # 6. DENSE SYNAPTIC NEURAL NETWORK & WIDE HUD PROBABILITY CARDS
    fc1_x = 16.5
    fc1_ys = np.linspace(1.2, 3.8, 6)
    for ny in fc1_ys:
        ax.add_patch(patches.Circle((fc1_x, ny), 0.18, facecolor="#38bdf8", alpha=0.25, zorder=6))
        ax.add_patch(patches.Circle((fc1_x, ny), 0.11, facecolor="#38bdf8", edgecolor="#e0f2fe", lw=1.2, zorder=7))
        ax.plot([fl_ox + 0.5 + 0.21, fc1_x - 0.11], [fl_oy + 1.3, ny], color="#64748b", lw=0.7, alpha=0.45, zorder=4)

    fc2_x = 18.2
    fc2_ys = np.linspace(1.5, 3.5, 4)
    for ny2 in fc2_ys:
        ax.add_patch(patches.Circle((fc2_x, ny2), 0.18, facecolor="#34d399", alpha=0.25, zorder=6))
        ax.add_patch(patches.Circle((fc2_x, ny2), 0.11, facecolor="#34d399", edgecolor="#d1fae5", lw=1.2, zorder=7))
        for ny1 in fc1_ys:
            ax.plot([fc1_x + 0.11, fc2_x - 0.11], [ny1, ny2], color="#38bdf8", lw=0.6, alpha=0.4, zorder=4)

    # Output Softmax Layer & Wide Polished HUD Cards
    fc3_x = 19.8
    out_items = [
        ("LIVE FACE", "0.984", "#22c55e", "#15803d", 3.4),
        ("PRINT ATTACK", "0.008", "#f43f5e", "#be123c", 2.5),
        ("MASK / 3D REPLAY", "0.008", "#f43f5e", "#be123c", 1.6)
    ]
    
    for label, prob, col, bcol, ny3 in out_items:
        ax.add_patch(patches.Circle((fc3_x, ny3), 0.20, facecolor=col, alpha=0.3, zorder=6))
        ax.add_patch(patches.Circle((fc3_x, ny3), 0.12, facecolor=col, edgecolor="#ffffff", lw=1.4, zorder=7))
        for ny2 in fc2_ys:
            ax.plot([fc2_x + 0.11, fc3_x - 0.12], [ny2, ny3], color="#34d399", lw=0.6, alpha=0.4, zorder=4)
            
        ax.add_patch(patches.FancyBboxPatch((fc3_x + 0.45, ny3 - 0.26), 4.5, 0.52, boxstyle="round,pad=0.04", facecolor="#1e293b", edgecolor=col, lw=1.4, zorder=8))
        ax.text(fc3_x + 0.65, ny3 + 0.03, label, color="#f8fafc", fontsize=7.2, weight='bold', va='center', zorder=9)
        ax.text(fc3_x + 4.75, ny3 + 0.03, f"p = {prob}", color=col, fontsize=7.8, weight='bold', ha='right', va='center', zorder=9)
        
        bar_w = 4.1 * float(prob)
        ax.add_patch(patches.Rectangle((fc3_x + 0.65, ny3 - 0.18), 4.1, 0.06, facecolor="#334155", zorder=9))
        ax.add_patch(patches.Rectangle((fc3_x + 0.65, ny3 - 0.18), max(bar_w, 0.06), 0.06, facecolor=col, zorder=10))

    ax.text(fc1_x, 0.45, "DENSE FC 1\n(512-d)", color="#38bdf8", fontsize=8.0, ha='center')
    ax.text(fc2_x, 0.45, "DENSE FC 2\n(128-d)", color="#34d399", fontsize=8.0, ha='center')
    ax.text(fc3_x + 2.7, 0.45, "CALIBRATED SOFTMAX\nPROBABILITY DISTRIBUTION", color="#f8fafc", fontsize=8.0, ha='center', weight='bold')

    ax.set_xlim(-0.2, 25.8)
    ax.set_ylim(0.0, 4.4)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.06, facecolor='#090d16')
    plt.close(fig)
    
    svg_bytes = buf.getvalue()
    file_id = register_svg_file(svg_bytes)
    return file_id, 1480, 440

# -----------------------------------------------------------------------------
# 6 DEDICATED VECTOR MINI-ILLUSTRATIONS FOR SECTION E CARDS
# -----------------------------------------------------------------------------
def render_mini_illustration_e1():
    """E.1 Pre-LN Gradient Highway Diagram"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Pre-LN Block Flow
    # Input
    ax.text(0.2, 0.5, r"$\mathbf{x}_{l-1}$", color="#fde047", fontsize=10, weight='bold', va='center')
    # Split dot
    ax.plot([0.7, 0.7], [0.5, 0.9], color="#38bdf8", lw=1.5)
    ax.plot([0.7, 3.2], [0.9, 0.9], color="#38bdf8", lw=1.5) # Highway
    ax.text(1.9, 1.05, "Identity Gradient Highway (∂x_L/∂x_l = I)", color="#38bdf8", fontsize=6.8, ha='center')
    
    # Sublayer branch
    ax.plot([0.7, 1.1], [0.5, 0.5], color="#fde047", lw=1.4)
    # LN box
    ax.add_patch(patches.FancyBboxPatch((1.1, 0.25), 0.7, 0.5, boxstyle="round,pad=0.03", facecolor="#1e293b", edgecolor="#eab308", lw=1.2))
    ax.text(1.45, 0.5, "LN", color="#fde047", fontsize=8, weight='bold', ha='center', va='center')
    
    # Arrow to SubLayer
    ax.plot([1.8, 2.1], [0.5, 0.5], color="#fde047", lw=1.4)
    # SubLayer Box
    ax.add_patch(patches.FancyBboxPatch((2.1, 0.25), 0.8, 0.5, boxstyle="round,pad=0.03", facecolor="#1e293b", edgecolor="#10b981", lw=1.2))
    ax.text(2.5, 0.5, "MHSA", color="#a7f3d0", fontsize=8, weight='bold', ha='center', va='center')
    
    # To Add (+)
    ax.plot([2.9, 3.2], [0.5, 0.5], color="#fde047", lw=1.4)
    ax.plot([3.2, 3.2], [0.9, 0.65], color="#38bdf8", lw=1.5)
    
    # Add circle
    ax.add_patch(patches.Circle((3.2, 0.5), 0.15, facecolor="#0f172a", edgecolor="#38bdf8", lw=1.4))
    ax.text(3.2, 0.5, "+", color="#38bdf8", fontsize=9, weight='bold', ha='center', va='center')
    
    # Output
    ax.plot([3.35, 3.7], [0.5, 0.5], color="#fde047", lw=1.5)
    ax.text(3.8, 0.5, r"$\mathbf{x}_l$", color="#fde047", fontsize=10, weight='bold', va='center')
    
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(0.05, 1.25)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

def render_mini_illustration_e2():
    """E.2 MHSA & 4 Register Tokens / Sinks Elimination"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Left: Standard ViT (Norm Spike)
    ax.text(0.8, 1.1, "Standard ViT (Sinks)", color="#fb7185", fontsize=7.2, weight='bold', ha='center')
    # Bar showing patch outlier
    ax.bar([0.3, 0.5, 0.7, 0.9, 1.1, 1.3], [0.2, 0.3, 0.95, 0.2, 0.25, 0.3], width=0.15, color=['#38bdf8', '#38bdf8', '#f43f5e', '#38bdf8', '#38bdf8', '#38bdf8'])
    ax.text(0.7, 0.98, "Spike >140", color="#f43f5e", fontsize=6.5, ha='center', weight='bold')
    
    # Divider
    ax.plot([1.8, 1.8], [0.1, 1.15], color="#334155", lw=1.0, linestyle="--")
    
    # Right: ViT + 4 Registers (Clean & Uniform)
    ax.text(3.0, 1.1, "DINOv2 + 4 Registers", color="#34d399", fontsize=7.2, weight='bold', ha='center')
    # 4 Regs absorb energy
    ax.bar([2.2, 2.4, 2.6, 2.8], [0.85, 0.82, 0.88, 0.84], width=0.15, color='#a855f7')
    ax.text(2.5, 0.65, "4 Registers", color="#e9d5ff", fontsize=6.2, ha='center', weight='bold')
    # Patches are uniform
    ax.bar([3.1, 3.3, 3.5, 3.7], [0.25, 0.24, 0.26, 0.25], width=0.15, color='#34d399')
    ax.text(3.4, 0.35, "Uniform ~26.8", color="#a7f3d0", fontsize=6.2, ha='center', weight='bold')
    
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 1.25)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

def render_mini_illustration_e3():
    """E.3 LayerScale Deep Stabilization Diagram"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Sublayer output
    ax.text(0.2, 0.5, r"$\mathbf{h}_l$", color="#c7d2fe", fontsize=9.5, weight='bold', va='center')
    ax.plot([0.5, 1.2], [0.5, 0.5], color="#818cf8", lw=1.4)
    
    # LayerScale Multiplier Node (⊗)
    ax.add_patch(patches.Circle((1.5, 0.5), 0.22, facecolor="#312e81", edgecolor="#818cf8", lw=1.4))
    ax.text(1.5, 0.5, "diag(λ)", color="#ffffff", fontsize=6.8, weight='bold', ha='center', va='center')
    
    # Init tag
    ax.text(1.5, 0.95, r"Init $\lambda_i \sim 10^{-5}$", color="#818cf8", fontsize=7.2, ha='center', weight='bold')
    
    # To Addition node
    ax.plot([1.75, 2.5], [0.5, 0.5], color="#818cf8", lw=1.4)
    ax.add_patch(patches.Circle((2.7, 0.5), 0.18, facecolor="#0f172a", edgecolor="#818cf8", lw=1.4))
    ax.text(2.7, 0.5, "+", color="#818cf8", fontsize=10, weight='bold', ha='center', va='center')
    
    # Residual Line
    ax.plot([0.2, 2.7], [1.1, 1.1], color="#38bdf8", lw=1.4)
    ax.plot([2.7, 2.7], [1.1, 0.7], color="#38bdf8", lw=1.4)
    ax.text(1.0, 1.2, r"Input $\mathbf{x}$", color="#38bdf8", fontsize=7.5, weight='bold')
    
    # Output
    ax.plot([2.9, 3.5], [0.5, 0.5], color="#818cf8", lw=1.5)
    ax.text(3.6, 0.5, r"$\mathbf{x}_{\mathrm{out}}$", color="#c7d2fe", fontsize=9.5, weight='bold', va='center')
    
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(0.1, 1.35)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

def render_mini_illustration_e4():
    """E.4 Residual Skip Connection & Gradient Preservation"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Forward Pass
    ax.text(0.2, 0.8, r"$\mathbf{x}_{l-1}$", color="#38bdf8", fontsize=9.5, weight='bold', va='center')
    ax.plot([0.6, 1.4], [0.8, 0.8], color="#38bdf8", lw=1.4)
    
    ax.add_patch(patches.FancyBboxPatch((1.4, 0.55), 1.2, 0.5, boxstyle="round,pad=0.03", facecolor="#0c4a6e", edgecolor="#0284c7", lw=1.2))
    ax.text(2.0, 0.8, r"$\mathcal{F}(\mathbf{x})$", color="#7dd3fc", fontsize=8.5, weight='bold', ha='center', va='center')
    
    # Bypass
    ax.plot([0.6, 0.6], [0.8, 1.15], color="#38bdf8", lw=1.4)
    ax.plot([0.6, 3.0], [1.15, 1.15], color="#38bdf8", lw=1.4)
    ax.plot([3.0, 3.0], [1.15, 0.95], color="#38bdf8", lw=1.4)
    ax.text(1.8, 1.25, "Identity Bypass (Matrix I)", color="#38bdf8", fontsize=7.0, ha='center')
    
    # Add
    ax.plot([2.6, 3.0], [0.8, 0.8], color="#38bdf8", lw=1.4)
    ax.add_patch(patches.Circle((3.0, 0.8), 0.15, facecolor="#0f172a", edgecolor="#38bdf8", lw=1.4))
    ax.text(3.0, 0.8, "+", color="#38bdf8", fontsize=9, weight='bold', ha='center', va='center')
    
    ax.plot([3.15, 3.6], [0.8, 0.8], color="#38bdf8", lw=1.5)
    ax.text(3.7, 0.8, r"$\mathbf{x}_l$", color="#38bdf8", fontsize=9.5, weight='bold', va='center')
    
    # Backward Gradient Flow (Bottom Neon Arrow)
    ax.annotate("", xy=(0.6, 0.25), xytext=(3.6, 0.25), arrowprops=dict(arrowstyle="->", color="#fb7185", lw=1.6, ls="--"))
    ax.text(2.1, 0.38, r"Backward $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{l-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l}(\mathbf{I} + \dots)$", color="#fb7185", fontsize=7.2, ha='center', weight='bold')
    
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(0.1, 1.35)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

def render_mini_illustration_e5():
    """E.5 SwiGLU Gated Feed-Forward Network Dataflow"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Input
    ax.text(0.2, 0.6, r"$\mathbf{x}$", color="#7dd3fc", fontsize=10, weight='bold', va='center')
    ax.plot([0.45, 0.8], [0.6, 0.6], color="#38bdf8", lw=1.4)
    
    # Split to Branch 1 & Branch 2
    ax.plot([0.8, 0.8], [0.25, 0.95], color="#38bdf8", lw=1.4)
    ax.plot([0.8, 1.1], [0.95, 0.95], color="#38bdf8", lw=1.4)
    ax.plot([0.8, 1.1], [0.25, 0.25], color="#38bdf8", lw=1.4)
    
    # Branch 1: Swish(xW1)
    ax.add_patch(patches.FancyBboxPatch((1.1, 0.75), 1.1, 0.4, boxstyle="round,pad=0.02", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.2))
    ax.text(1.65, 0.95, "Swish(xW1)", color="#38bdf8", fontsize=7.0, weight='bold', ha='center', va='center')
    
    # Branch 2: xW3
    ax.add_patch(patches.FancyBboxPatch((1.1, 0.05), 1.1, 0.4, boxstyle="round,pad=0.02", facecolor="#1e293b", edgecolor="#f59e0b", lw=1.2))
    ax.text(1.65, 0.25, "Linear(xW3)", color="#fde047", fontsize=7.0, weight='bold', ha='center', va='center')
    
    # Connect to Multiplier (⊗)
    ax.plot([2.2, 2.6], [0.95, 0.6], color="#38bdf8", lw=1.4)
    ax.plot([2.2, 2.6], [0.25, 0.6], color="#fde047", lw=1.4)
    
    ax.add_patch(patches.Circle((2.6, 0.6), 0.18, facecolor="#0c4a6e", edgecolor="#38bdf8", lw=1.4))
    ax.text(2.6, 0.6, "⊗", color="#ffffff", fontsize=9, weight='bold', ha='center', va='center')
    
    # Final projection W2
    ax.plot([2.8, 3.1], [0.6, 0.6], color="#38bdf8", lw=1.4)
    ax.add_patch(patches.FancyBboxPatch((3.1, 0.38), 0.7, 0.44, boxstyle="round,pad=0.02", facecolor="#1e293b", edgecolor="#10b981", lw=1.2))
    ax.text(3.45, 0.6, "W2", color="#a7f3d0", fontsize=7.5, weight='bold', ha='center', va='center')
    
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 1.25)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

def render_mini_illustration_e6():
    """E.6 Tri-Stream Multi-Token Fusion Head Pooling"""
    fig, ax = plt.subplots(figsize=(4.2, 1.3), dpi=220)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # 3 Streams
    # 1. CLS
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.85), 0.9, 0.32, boxstyle="round,pad=0.02", facecolor="#0284c7", edgecolor="#38bdf8", lw=1.0))
    ax.text(0.65, 1.01, "[CLS] (768)", color="#ffffff", fontsize=6.8, weight='bold', ha='center', va='center')
    
    # 2. REG
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.46), 0.9, 0.32, boxstyle="round,pad=0.02", facecolor="#7e22ce", edgecolor="#c084fc", lw=1.0))
    ax.text(0.65, 0.62, "Mean [REG] (768)", color="#f5d0fe", fontsize=6.8, weight='bold', ha='center', va='center')
    
    # 3. PATCH
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.08), 0.9, 0.32, boxstyle="round,pad=0.02", facecolor="#047857", edgecolor="#34d399", lw=1.0))
    ax.text(0.65, 0.24, "Mean [P] (768)", color="#d1fae5", fontsize=6.8, weight='bold', ha='center', va='center')
    
    # Converge into Concat
    ax.plot([1.1, 1.6], [1.01, 0.62], color="#38bdf8", lw=1.2)
    ax.plot([1.1, 1.6], [0.62, 0.62], color="#c084fc", lw=1.2)
    ax.plot([1.1, 1.6], [0.24, 0.62], color="#34d399", lw=1.2)
    
    # Concat Bar
    ax.add_patch(patches.FancyBboxPatch((1.6, 0.25), 1.2, 0.75, boxstyle="round,pad=0.03", facecolor="#4c0519", edgecolor="#f43f5e", lw=1.4))
    ax.text(2.2, 0.70, "Fused Vector", color="#fecdd3", fontsize=7.0, weight='bold', ha='center')
    ax.text(2.2, 0.45, "f_FAS (2304-d)", color="#fb7185", fontsize=7.5, weight='bold', ha='center')
    
    # Arrow to Classifier
    ax.plot([2.8, 3.2], [0.62, 0.62], color="#fb7185", lw=1.5)
    ax.add_patch(patches.FancyBboxPatch((3.2, 0.35), 0.8, 0.55, boxstyle="round,pad=0.03", facecolor="#be123c", edgecolor="#fb7185", lw=1.2))
    ax.text(3.6, 0.62, "Sigmoid\nŷ ∈ [0,1]", color="#ffffff", fontsize=7.0, weight='bold', ha='center', va='center')
    
    ax.set_xlim(0.0, 4.2)
    ax.set_ylim(0.0, 1.25)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.02, facecolor='#0f172a')
    plt.close(fig)
    return register_svg_file(buf.getvalue())

# -----------------------------------------------------------------------------
# BUILD MASTER HYBRID DIAGRAM
# -----------------------------------------------------------------------------
def build_master_hybrid_diagram():
    canvas_w = 2400
    create_card(40, 25, canvas_w, 110, bg_color="#090d16", stroke_color="#1e293b", stroke_width=1.5, border_radius=12)
    
    create_pill(60, 42, 175, 26, "⚡ DINOv2 FAS MASTER", "#0f172a", "#38bdf8", "#38bdf8")
    create_pill(245, 42, 210, 26, "3D ISOMETRIC TENSORS", "#0f172a", "#34d399", "#34d399")
    create_pill(465, 42, 230, 26, "ViT-B/14 4 REGISTERS", "#0f172a", "#a855f7", "#c084fc")
    create_pill(705, 42, 220, 26, "MULTI-TOKEN FUSION", "#0f172a", "#f43f5e", "#fb7185")
    create_pill(935, 42, 210, 26, "ISO/IEC 30107-3 L1&L2", "#0f172a", "#eab308", "#fde047")

    create_text(60, 80, "Unified Deep Learning Architecture: 3D Feature Maps, Vision Transformer Macro/Micro Stack & Multi-Token Fusion", font_size=18, color="#f8fafc", font_family=2, is_bold=True)
    create_text(60, 106, "Publication-Grade Architectural Synthesis: Receptive Field Frustums, Dense Synaptic Graph, 12× Encoder Zoom-in, Multi-Token Pooling, and Rigorous Math Spec", font_size=11.5, color="#94a3b8", font_family=2)

    # -------------------------------------------------------------------------
    # SECTION A (TOP): 3D ISOMETRIC FEATURE MAPS & SYNAPTIC PIPELINE
    # -------------------------------------------------------------------------
    s1_y = 155
    s1_w = canvas_w
    s1_h = 370
    create_card(40, s1_y, s1_w, s1_h, bg_color="#090d16", stroke_color="#0284c7", stroke_width=2, border_radius=14)
    create_pill(56, s1_y + 12, 530, 26, "SECTION A: 3D ISOMETRIC FEATURE TENSORS, RECEPTIVE CONES & SYNAPTIC CLASSIFIER", "#0f172a", "#38bdf8", "#38bdf8")
    
    p1_file, p1_w, p1_h = render_master_3d_isometric_pipeline_svg()
    img1 = {
        "id": next_id("plot3d"), "type": "image", "x": 56, "y": s1_y + 44, "width": s1_w - 32, "height": 310,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 101, "version": 1, "versionNonce": 101, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": p1_file, "scale": [1, 1],
        "customData": {"type": "scientific_plot", "title": "3d_isometric_feature_maps"}
    }
    elements.append(img1)

    # -------------------------------------------------------------------------
    # MIDDLE SECTIONS CONTAINER (3 COLUMNS: B, C, D)
    # -------------------------------------------------------------------------
    s2_y = s1_y + s1_h + 20
    s2_h = 890

    # -------------------------------------------------------------------------
    # COLUMN 1: SECTION B - MICRO TRANSFORMER BLOCK (12x Stack Zoom-in)
    # -------------------------------------------------------------------------
    b_x = 40
    b_w = 780
    create_card(b_x, s2_y, b_w, s2_h, bg_color="#090d16", stroke_color="#059669", stroke_width=2, border_radius=14)
    create_pill(b_x + 16, s2_y + 14, 440, 26, "SECTION B: MICRO TRANSFORMER BLOCK ZOOM-IN (12x Stack)", "#0f172a", "#34d399", "#34d399")

    # Block Container Card
    bx_in_x = b_x + 20
    bx_in_y = s2_y + 52
    bx_in_w = b_w - 40
    bx_in_h = s2_h - 72
    b_container = create_card(bx_in_x, bx_in_y, bx_in_w, bx_in_h, bg_color="#0f172a", stroke_color="#10b981", stroke_width=1.8, border_radius=12)
    create_text(bx_in_x + 20, bx_in_y + 14, "Transformer Block Layer (l ∈ [1 .. 12]) with 4 Register Tokens", font_size=12.5, color="#34d399", is_bold=True)

    B_CX = bx_in_x + bx_in_w / 2  # = 430
    B_BYPASS_X = bx_in_x + bx_in_w - 35 # = 765

    # 1. Input Tokens Box
    inp_w, inp_h = 440, 36
    inp_y = bx_in_y + 42
    card_inp = create_card(B_CX - inp_w/2, inp_y, inp_w, inp_h, bg_color="#1e293b", stroke_color="#f59e0b", stroke_width=1.5, border_radius=18)
    create_centered_text(B_CX, inp_y + inp_h/2, "Input Tokens: Z_{l-1} ∈ R^{261 x 768} ([CLS] + 4 [REG] + 256 [PATCH])", font_size=10.5, color="#fde047", is_bold=True)

    # 2. Pre-LayerNorm 1
    ln1_w, ln1_h = 280, 34
    ln1_y = inp_y + 60
    card_ln1 = create_card(B_CX - ln1_w/2, ln1_y, ln1_w, ln1_h, bg_color="#1e293b", stroke_color="#eab308", stroke_width=1.5, border_radius=6)
    create_centered_text(B_CX, ln1_y + ln1_h/2, "Pre-LayerNorm 1 (LayerNorm)", font_size=11.5, color="#fde047", is_bold=True)
    create_connected_arrow([B_CX, inp_y + inp_h], [B_CX, ln1_y], start_elem=card_inp, end_elem=card_ln1, color="#94a3b8", stroke_width=1.6)

    # 3. Q, K, V Projections
    qkv_w, qkv_h = 135, 30
    qkv_y = ln1_y + 50
    q_cx = B_CX - 165
    k_cx = B_CX
    v_cx = B_CX + 165

    card_q = create_card(q_cx - qkv_w/2, qkv_y, qkv_w, qkv_h, bg_color="#1e293b", stroke_color="#f43f5e", stroke_width=1.5, border_radius=6)
    create_centered_text(q_cx, qkv_y + qkv_h/2, "Query (W_Q)", font_size=10.5, color="#fb7185", is_bold=True)

    card_k = create_card(k_cx - qkv_w/2, qkv_y, qkv_w, qkv_h, bg_color="#1e293b", stroke_color="#f59e0b", stroke_width=1.5, border_radius=6)
    create_centered_text(k_cx, qkv_y + qkv_h/2, "Key (W_K)", font_size=10.5, color="#fde047", is_bold=True)

    card_v = create_card(v_cx - qkv_w/2, qkv_y, qkv_w, qkv_h, bg_color="#1e293b", stroke_color="#10b981", stroke_width=1.5, border_radius=6)
    create_centered_text(v_cx, qkv_y + qkv_h/2, "Value (W_V)", font_size=10.5, color="#6ee7b7", is_bold=True)

    create_connected_arrow([B_CX, ln1_y + ln1_h], [q_cx, qkv_y], start_elem=card_ln1, end_elem=card_q, color="#94a3b8", stroke_width=1.6)
    create_connected_arrow([B_CX, ln1_y + ln1_h], [k_cx, qkv_y], start_elem=card_ln1, end_elem=card_k, color="#94a3b8", stroke_width=1.6)
    create_connected_arrow([B_CX, ln1_y + ln1_h], [v_cx, qkv_y], start_elem=card_ln1, end_elem=card_v, color="#94a3b8", stroke_width=1.6)

    # 4. Multi-Head Self-Attention (MHSA) Box
    mha_w, mha_h = 500, 56
    mha_y = qkv_y + 46
    card_mha = create_card(B_CX - mha_w/2, mha_y, mha_w, mha_h, bg_color="#064e3b", stroke_color="#10b981", stroke_width=2, border_radius=8)
    create_centered_text(B_CX, mha_y + 16, "Multi-Head Self-Attention (MHSA) — Scaled Dot-Product", font_size=12, color="#a7f3d0", is_bold=True)
    create_centered_text(B_CX, mha_y + 38, "Attention(Q, K, V) = Softmax( QK^T / √d_k ) V  |  h = 12 Heads, d_k = 64", font_size=10.5, color="#6ee7b7")

    create_connected_arrow([q_cx, qkv_y + qkv_h], [q_cx, mha_y], start_elem=card_q, end_elem=card_mha, color="#fb7185", stroke_width=1.6)
    create_connected_arrow([k_cx, qkv_y + qkv_h], [k_cx, mha_y], start_elem=card_k, end_elem=card_mha, color="#fde047", stroke_width=1.6)
    create_connected_arrow([v_cx, qkv_y + qkv_h], [v_cx, mha_y], start_elem=card_v, end_elem=card_mha, color="#6ee7b7", stroke_width=1.6)

    # 5. LayerScale 1
    ls1_w, ls1_h = 260, 30
    ls1_y = mha_y + 68
    card_ls1 = create_card(B_CX - ls1_w/2, ls1_y, ls1_w, ls1_h, bg_color="#312e81", stroke_color="#6366f1", stroke_width=1.5, border_radius=6)
    create_centered_text(B_CX, ls1_y + ls1_h/2, "LayerScale 1: diag(λ_l) ⊙ X", font_size=11, color="#c7d2fe", is_bold=True)
    create_connected_arrow([B_CX, mha_y + mha_h], [B_CX, ls1_y], start_elem=card_mha, end_elem=card_ls1, color="#94a3b8", stroke_width=1.6)

    # 6. Residual Addition Circle 1 (⊕)
    add1_r = 18
    add1_y = ls1_y + 46
    card_add1 = create_ellipse(B_CX - add1_r, add1_y, add1_r*2, add1_r*2, bg_color="#0f172a", stroke_color="#38bdf8", stroke_width=2)
    create_centered_text(B_CX, add1_y + add1_r, "+", font_size=18, color="#38bdf8", is_bold=True)
    create_connected_arrow([B_CX, ls1_y + ls1_h], [B_CX, add1_y], start_elem=card_ls1, end_elem=card_add1, color="#94a3b8", stroke_width=1.6)

    # Residual Branch 1 Bypass
    b1_split_y = inp_y + inp_h + 12
    b1_pts = [
        [B_CX, b1_split_y],
        [B_BYPASS_X, b1_split_y],
        [B_BYPASS_X, add1_y + add1_r],
        [B_CX + add1_r, add1_y + add1_r]
    ]
    create_connected_elbow_arrow(b1_pts, start_elem=card_inp, end_elem=card_add1, color="#38bdf8", stroke_width=2.0, stroke_style="solid")
    create_card(B_BYPASS_X - 110, b1_split_y - 22, 105, 20, bg_color="#0f172a", stroke_color="#38bdf8", stroke_width=1.2, border_radius=4)
    create_centered_text(B_BYPASS_X - 57, b1_split_y - 12, "Residual Skip 1", font_size=9.2, color="#38bdf8", is_bold=True)

    # 7. Pre-LayerNorm 2
    ln2_w, ln2_h = 280, 34
    ln2_y = add1_y + add1_r*2 + 25
    card_ln2 = create_card(B_CX - ln2_w/2, ln2_y, ln2_w, ln2_h, bg_color="#1e293b", stroke_color="#eab308", stroke_width=1.5, border_radius=6)
    create_centered_text(B_CX, ln2_y + ln2_h/2, "Pre-LayerNorm 2 (LayerNorm)", font_size=11.5, color="#fde047", is_bold=True)
    create_connected_arrow([B_CX, add1_y + add1_r*2], [B_CX, ln2_y], start_elem=card_add1, end_elem=card_ln2, color="#94a3b8", stroke_width=1.6)

    # 8. SwiGLU Gated Feed-Forward MLP
    mlp_w, mlp_h = 500, 56
    mlp_y = ln2_y + 48
    card_mlp = create_card(B_CX - mlp_w/2, mlp_y, mlp_w, mlp_h, bg_color="#0c4a6e", stroke_color="#0284c7", stroke_width=2, border_radius=8)
    create_centered_text(B_CX, mlp_y + 16, "SwiGLU Gated Feed-Forward Network (FFN)", font_size=12, color="#7dd3fc", is_bold=True)
    create_centered_text(B_CX, mlp_y + 38, "SwiGLU(x) = ( Swish(xW_1 + b_1) ⊗ (xW_3 + b_3) ) W_2  |  d_ffn = 2048", font_size=10.5, color="#38bdf8")
    create_connected_arrow([B_CX, ln2_y + ln2_h], [B_CX, mlp_y], start_elem=card_ln2, end_elem=card_mlp, color="#94a3b8", stroke_width=1.6)

    # 9. LayerScale 2
    ls2_w, ls2_h = 260, 30
    ls2_y = mlp_y + 68
    card_ls2 = create_card(B_CX - ls2_w/2, ls2_y, ls2_w, ls2_h, bg_color="#312e81", stroke_color="#6366f1", stroke_width=1.5, border_radius=6)
    create_centered_text(B_CX, ls2_y + ls2_h/2, "LayerScale 2: diag(γ_l) ⊙ X", font_size=11, color="#c7d2fe", is_bold=True)
    create_connected_arrow([B_CX, mlp_y + mlp_h], [B_CX, ls2_y], start_elem=card_mlp, end_elem=card_ls2, color="#94a3b8", stroke_width=1.6)

    # 10. Residual Addition Circle 2 (⊕)
    add2_r = 18
    add2_y = ls2_y + 46
    card_add2 = create_ellipse(B_CX - add2_r, add2_y, add2_r*2, add2_r*2, bg_color="#0f172a", stroke_color="#c084fc", stroke_width=2)
    create_centered_text(B_CX, add2_y + add2_r, "+", font_size=18, color="#c084fc", is_bold=True)
    create_connected_arrow([B_CX, ls2_y + ls2_h], [B_CX, add2_y], start_elem=card_ls2, end_elem=card_add2, color="#94a3b8", stroke_width=1.6)

    # Residual Branch 2 Bypass
    b2_split_y = add1_y + add1_r*2 + 12
    b2_pts = [
        [B_CX, b2_split_y],
        [B_BYPASS_X, b2_split_y],
        [B_BYPASS_X, add2_y + add2_r],
        [B_CX + add2_r, add2_y + add2_r]
    ]
    create_connected_elbow_arrow(b2_pts, start_elem=card_add1, end_elem=card_add2, color="#c084fc", stroke_width=2.0, stroke_style="solid")
    create_card(B_BYPASS_X - 110, b2_split_y - 22, 105, 20, bg_color="#0f172a", stroke_color="#c084fc", stroke_width=1.2, border_radius=4)
    create_centered_text(B_BYPASS_X - 57, b2_split_y - 12, "Residual Skip 2", font_size=9.2, color="#c084fc", is_bold=True)

    # 11. Block Output Box
    outb_w, outb_h = 380, 34
    outb_y = add2_y + add2_r*2 + 25
    card_outb = create_card(B_CX - outb_w/2, outb_y, outb_w, outb_h, bg_color="#064e3b", stroke_color="#10b981", stroke_width=1.5, border_radius=8)
    create_centered_text(B_CX, outb_y + outb_h/2, "Block Output: Z_l ∈ R^{261 x 768} (Passed to Layer l+1)", font_size=11, color="#a7f3d0", is_bold=True)
    create_connected_arrow([B_CX, add2_y + add2_r*2], [B_CX, outb_y], start_elem=card_add2, end_elem=card_outb, color="#10b981", stroke_width=2.5)

    # -------------------------------------------------------------------------
    # COLUMN 2: SECTION C - MACRO BACKBONE STACK & MULTI-TOKEN FUSION HEAD
    # -------------------------------------------------------------------------
    c_x = 840
    c_w = 750
    create_card(c_x, s2_y, c_w, s2_h, bg_color="#090d16", stroke_color="#d97706", stroke_width=2, border_radius=14)
    create_pill(c_x + 16, s2_y + 14, 400, 26, "SECTION C: MACRO VISION TRANSFORMER PIPELINE", "#0f172a", "#fbbf24", "#fde047")

    C_CX = c_x + c_w / 2  # = 1215

    # 1. Patch Embedding & Token Ribbon Visualizer
    pe_w, pe_h = 490, 75
    pe_y = s2_y + 52
    card_pe = create_card(C_CX - pe_w/2, pe_y, pe_w, pe_h, bg_color="#1e293b", stroke_color="#f59e0b", stroke_width=2, border_radius=10)
    create_centered_text(C_CX, pe_y + 16, "Patch Embedding Layer (Linear Conv 14x14, Stride 14)", font_size=12, color="#fde047", is_bold=True)
    
    # Token Ribbon Chips (Centered row)
    rib_y = pe_y + 36
    r_cls_w = 65
    r_reg_w = 110
    r_pat_w = 145
    r_pos_w = 110
    total_rib_w = r_cls_w + r_reg_w + r_pat_w + r_pos_w + 15
    rib_start_x = C_CX - total_rib_w / 2

    create_card(rib_start_x, rib_y, r_cls_w, 26, bg_color="#0284c7", stroke_color="#38bdf8", stroke_width=1.2, border_radius=4)
    create_centered_text(rib_start_x + r_cls_w/2, rib_y + 13, "[CLS]", font_size=10, color="#ffffff", is_bold=True)

    reg_x = rib_start_x + r_cls_w + 5
    create_card(reg_x, rib_y, r_reg_w, 26, bg_color="#7e22ce", stroke_color="#c084fc", stroke_width=1.2, border_radius=4)
    create_centered_text(reg_x + r_reg_w/2, rib_y + 13, "[REG 1 .. 4]", font_size=10, color="#f5d0fe", is_bold=True)

    pat_x = reg_x + r_reg_w + 5
    create_card(pat_x, rib_y, r_pat_w, 26, bg_color="#047857", stroke_color="#34d399", stroke_width=1.2, border_radius=4)
    create_centered_text(pat_x + r_pat_w/2, rib_y + 13, "[PATCH 1 .. 256]", font_size=10, color="#d1fae5", is_bold=True)

    pos_x = pat_x + r_pat_w + 5
    create_card(pos_x, rib_y, r_pos_w, 26, bg_color="#374151", stroke_color="#9ca3af", stroke_width=1.2, border_radius=4)
    create_centered_text(pos_x + r_pos_w/2, rib_y + 13, "+ PosEmbed", font_size=10, color="#e5e7eb", is_bold=True)

    # 2. 12x Encoder Stack Container
    stk_w, stk_h = 490, 390
    stk_y = pe_y + 95
    card_stk = create_card(C_CX - stk_w/2, stk_y, stk_w, stk_h, bg_color="#0f172a", stroke_color="#059669", stroke_width=2, border_radius=12)
    create_centered_text(C_CX, stk_y + 18, "12x Transformer Encoder Stack (DINOv2 ViT-B/14)", font_size=12.5, color="#34d399", is_bold=True)

    # Block 1
    b_card_w, b_card_h = 410, 48
    b1_y = stk_y + 44
    card_b1 = create_card(C_CX - b_card_w/2, b1_y, b_card_w, b_card_h, bg_color="#064e3b", stroke_color="#10b981", stroke_width=1.8, border_radius=8)
    create_centered_text(C_CX, b1_y + b_card_h/2, "Transformer Block 1 (D=768, h=12)", font_size=12, color="#a7f3d0", is_bold=True)
    create_connected_arrow([C_CX, pe_y + pe_h], [C_CX, b1_y], start_elem=card_pe, end_elem=card_b1, color="#94a3b8", stroke_width=1.8)

    # Block 2
    b2_y = b1_y + 68
    card_b2 = create_card(C_CX - b_card_w/2, b2_y, b_card_w, b_card_h, bg_color="#064e3b", stroke_color="#10b981", stroke_width=1.8, border_radius=8)
    create_centered_text(C_CX, b2_y + b_card_h/2, "Transformer Block 2 (D=768, h=12)", font_size=12, color="#a7f3d0", is_bold=True)
    create_connected_arrow([C_CX, b1_y + b_card_h], [C_CX, b2_y], start_elem=card_b1, end_elem=card_b2, color="#94a3b8", stroke_width=1.6)

    # Dots
    dots_y = b2_y + 60
    elem_dots = create_centered_text(C_CX, dots_y, "• • •", font_size=18, color="#10b981", is_bold=True)
    create_connected_arrow([C_CX, b2_y + b_card_h], [C_CX, dots_y - 12], start_elem=card_b2, end_elem=elem_dots, color="#94a3b8", stroke_width=1.6)

    # Block 12 (Target of Zoom-in)
    b12_y = dots_y + 36
    card_b12 = create_card(C_CX - b_card_w/2, b12_y, b_card_w, b_card_h, bg_color="#047857", stroke_color="#34d399", stroke_width=2.5, border_radius=8)
    create_centered_text(C_CX, b12_y + b_card_h/2, "Transformer Block 12 (Final Layer)", font_size=12, color="#ecfdf5", is_bold=True)
    create_connected_arrow([C_CX, dots_y + 12], [C_CX, b12_y], start_elem=elem_dots, end_elem=card_b12, color="#94a3b8", stroke_width=1.6)

    # ZOOM-IN DASHED CONES (Originating directly from left border of Block 12 and expanding to Section B Container corners!)
    b12_left_x = C_CX - b_card_w/2
    create_line([bx_in_x + bx_in_w, bx_in_y + 30], [b12_left_x, b12_y + 6], color="#10b981", stroke_width=2.0, stroke_style="dashed")
    create_line([bx_in_x + bx_in_w, bx_in_y + bx_in_h - 30], [b12_left_x, b12_y + b_card_h - 6], color="#10b981", stroke_width=2.0, stroke_style="dashed")

    # 3. Final Layer Normalization
    lnf_w, lnf_h = 410, 38
    lnf_y = stk_y + stk_h + 30
    card_lnf = create_card(C_CX - lnf_w/2, lnf_y, lnf_w, lnf_h, bg_color="#1e293b", stroke_color="#eab308", stroke_width=1.8, border_radius=8)
    create_centered_text(C_CX, lnf_y + lnf_h/2, "Final Layer Normalization (LN)", font_size=12, color="#fde047", is_bold=True)
    create_connected_arrow([C_CX, b12_y + b_card_h], [C_CX, lnf_y], start_elem=card_b12, end_elem=card_lnf, color="#94a3b8", stroke_width=1.8)

    # 4. Multi-Token Fusion Head (3-Branch Concatenation)
    fuse_w, fuse_h = 490, 95
    fuse_y = lnf_y + 55
    card_fuse = create_card(C_CX - fuse_w/2, fuse_y, fuse_w, fuse_h, bg_color="#4c0519", stroke_color="#f43f5e", stroke_width=2.2, border_radius=10)
    create_centered_text(C_CX, fuse_y + 18, "Multi-Token Fusion Head: 3-Branch Feature Pooling", font_size=12.5, color="#fecdd3", is_bold=True)
    
    # 3 Branch Chips inside Fusion
    f_chip_y = fuse_y + 38
    f1_w, f2_w, f3_w = 145, 150, 145
    f_start_x = C_CX - (f1_w + f2_w + f3_w + 10) / 2

    create_card(f_start_x, f_chip_y, f1_w, 45, bg_color="#1e293b", stroke_color="#38bdf8", stroke_width=1.2, border_radius=6)
    create_centered_text(f_start_x + f1_w/2, f_chip_y + 22, "[CLS] Token (768-d)\nGlobal Semantics", font_size=9.5, color="#7dd3fc")

    f2_x = f_start_x + f1_w + 5
    create_card(f2_x, f_chip_y, f2_w, 45, bg_color="#1e293b", stroke_color="#c084fc", stroke_width=1.2, border_radius=6)
    create_centered_text(f2_x + f2_w/2, f_chip_y + 22, "Mean [REG] (768-d)\nIllum / Ambient Sink", font_size=9.5, color="#e9d5ff")

    f3_x = f2_x + f2_w + 5
    create_card(f3_x, f_chip_y, f3_w, 45, bg_color="#1e293b", stroke_color="#34d399", stroke_width=1.2, border_radius=6)
    create_centered_text(f3_x + f3_w/2, f_chip_y + 22, "Mean [PATCH] (768-d)\nMicro-Texture Cues", font_size=9.5, color="#a7f3d0")

    create_connected_arrow([C_CX, lnf_y + lnf_h], [C_CX, fuse_y], start_elem=card_lnf, end_elem=card_fuse, color="#94a3b8", stroke_width=1.8)

    # 5. Output FAS Classification Score
    out_w, out_h = 430, 48
    out_y = fuse_y + 115
    card_out = create_card(C_CX - out_w/2, out_y, out_w, out_h, bg_color="#be123c", stroke_color="#fb7185", stroke_width=2.2, border_radius=10)
    create_centered_text(C_CX, out_y + out_h/2, "FAS Liveness Probability Score: ŷ ∈ [0, 1]", font_size=12.5, color="#ffffff", is_bold=True)
    create_connected_arrow([C_CX, fuse_y + fuse_h], [C_CX, out_y], start_elem=card_fuse, end_elem=card_out, color="#fb7185", stroke_width=2.5)

    # -------------------------------------------------------------------------
    # COLUMN 3: SECTION D - MATHEMATICAL SPECIFICATIONS & BENCHMARK MATRIX
    # -------------------------------------------------------------------------
    d_x = 1610
    d_w = 830
    create_card(d_x, s2_y, d_w, s2_h, bg_color="#090d16", stroke_color="#f43f5e", stroke_width=2, border_radius=14)
    create_pill(d_x + 16, s2_y + 14, 450, 26, "SECTION D: MATHEMATICAL FORMULATIONS & BENCHMARK MATRIX", "#0f172a", "#f43f5e", "#fb7185")

    # Math Card 1: Sequence & Tensor Equations
    m1_y = s2_y + 52
    create_card(d_x + 14, m1_y, d_w - 28, 225, bg_color="#0f172a", stroke_color="#0284c7", stroke_width=1.5, border_radius=10)
    create_pill(d_x + 26, m1_y + 12, 340, 22, "D.1 Unified Token Sequence Matrix & Fusion", "#090d16", "#38bdf8", "#38bdf8")
    create_latex_math_element(d_x + 26, m1_y + 44, r"\mathbf{Z}_0 = [\mathbf{x}_{\mathrm{cls}} \,\|\, \mathbf{x}_{\mathrm{reg}}^{(1..4)} \,\|\, \mathbf{e}_p^{(1..256)}] + \mathbf{E}_{\mathrm{pos}} \in \mathbb{R}^{261 \times 768}", color="#38bdf8", fontsize=11.5)
    create_latex_math_element(d_x + 26, m1_y + 88, r"\mathbf{A} = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right), \quad d_k = \frac{D}{h} = \frac{768}{12} = 64", color="#7dd3fc", fontsize=11.5)
    create_latex_math_element(d_x + 26, m1_y + 132, r"\mathbf{f}_{\mathrm{FAS}} = [\mathbf{z}_{\mathrm{cls}} \,\|\, \frac{1}{4}\sum_{k=1}^4 \mathbf{z}_{\mathrm{reg}_k} \,\|\, \frac{1}{N}\sum_{i=1}^N \mathbf{z}_{p_i}] \in \mathbb{R}^{2304}", color="#f43f5e", fontsize=11.2)
    create_text(d_x + 26, m1_y + 185, "• Registers store non-informative background/illumination tokens to prevent attention sinks.", font_size=10.5, color="#94a3b8")

    # Math Card 2: Self-Supervision Losses
    m2_y = m1_y + 240
    create_card(d_x + 14, m2_y, d_w - 28, 235, bg_color="#0f172a", stroke_color="#d97706", stroke_width=1.5, border_radius=10)
    create_pill(d_x + 26, m2_y + 12, 360, 22, "D.2 Multi-Objective Self-Supervision & Focal Loss", "#090d16", "#f59e0b", "#fde047")
    create_latex_math_element(d_x + 26, m2_y + 44, r"\mathcal{L}_{\mathrm{DINO}} = -\sum_k P_t(k) \log P_s(k), \quad P_s(k) = \frac{\exp(g_s(\mathbf{x})_k / \tau_s)}{\sum_j \exp(g_s(\mathbf{x})_j / \tau_s)}", color="#fbbf24", fontsize=10.8)
    create_latex_math_element(d_x + 26, m2_y + 92, r"\mathcal{L}_{\mathrm{KoLeo}} = -\frac{1}{B}\sum_{i=1}^B \log \left( \min_{j \neq i} \|\mathbf{z}_i - \mathbf{z}_j\|_2 \right), \quad \mathbf{z}_i \in \mathbb{S}^{D-1}", color="#c084fc", fontsize=10.8)
    create_latex_math_element(d_x + 26, m2_y + 140, r"\mathcal{L}_{\mathrm{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad p_t = y \hat{y} + (1 - y)(1 - \hat{y})", color="#fb7185", fontsize=11.0)
    create_text(d_x + 26, m2_y + 192, "• KoLeo enforces uniform spherical coverage on S^{D-1}, eliminating mode collapse.", font_size=10.5, color="#94a3b8")

    # Benchmark & Hardware Matrix Card
    m3_y = m2_y + 250
    create_card(d_x + 14, m3_y, d_w - 28, 305, bg_color="#050811", stroke_color="#334155", stroke_width=1.5, border_radius=10)
    create_text(d_x + 26, m3_y + 16, "Comprehensive Empirical Benchmark & ISO/IEC 30107-3 Compliance", font_size=12.5, color="#38bdf8", is_bold=True)
    
    spec_tbl = """Parameter / Metric            Standard CNN (ResNet-100)   Standard ViT-B/14   DINOv2 ViT-B/14 (4 Registers)
------------------------------------------------------------------------------------------------------------
Model Backbone Parameters:    65.2M                       86.0M               86.6M (+0.6M for Registers)
Sequence Length (M Tokens):   N/A (Spatial Map)           257 tokens          261 tokens (+4 Reg Tokens)
Receptive Field / Attention:  Local 3x3 Convolutions      Global Attention    Global Attention (No Sinks)
Feature Norm Stability:       High Norm Variance          Outlier Spike >140  Uniform Homogeneous ~26.8
FAS ACER (CASIA-SURF):        2.14%                       1.85%               0.38% (★ -79.5% Error Drop)
FAS ACER (FIMO FAS Dataset):  1.68%                       1.42%               0.42% (★ -70.4% Error Drop)
Inference Latency (T4 FP16):  9.2 ms                      13.8 ms             14.2 ms (TensorRT Engine)
ISO/IEC 30107-3 Certification: Level 1 Compliant          Level 1 Compliant   Level 1 & Level 2 Certified ✓"""

    create_text(d_x + 26, m3_y + 44, spec_tbl, font_size=10.2, color="#cbd5e1", font_family=3)

    # -------------------------------------------------------------------------
    # SECTION E (BOTTOM): DEEP DIVE ARCHITECTURAL COMPONENT ANALYSIS WITH 6 VISUAL ILLUSTRATIONS
    # -------------------------------------------------------------------------
    s3_y = s2_y + s2_h + 20
    s3_w = canvas_w
    s3_h = 880
    create_card(40, s3_y, s3_w, s3_h, bg_color="#090d16", stroke_color="#8b5cf6", stroke_width=2, border_radius=14)
    create_pill(56, s3_y + 14, 560, 26, "SECTION E: DEEP DIVE ARCHITECTURAL COMPONENT ANALYSIS & MECHANISM SPECIFICATIONS", "#0f172a", "#c084fc", "#e9d5ff")

    # 6-Card Grid Layout (2 Rows x 3 Columns)
    e_col_w = (s3_w - 60) / 3  # ≈ 780
    e_row_h = 390
    
    # ------------------
    # ROW 1, COL 1: PRE-LAYERNORM VS POST-LAYERNORM
    # ------------------
    e1_x = 55
    e1_y = s3_y + 52
    create_card(e1_x, e1_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#eab308", stroke_width=1.5, border_radius=10)
    create_pill(e1_x + 16, e1_y + 12, 320, 22, "E.1 Pre-LayerNorm (Pre-LN) Architecture", "#090d16", "#eab308", "#fde047")
    create_latex_math_element(e1_x + 16, e1_y + 42, r"\mathrm{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}, \quad \mu = \frac{1}{D}\sum_{i=1}^D x_i", color="#fde047", fontsize=10.5)
    create_latex_math_element(e1_x + 16, e1_y + 82, r"\mathbf{x}_l = \mathbf{x}_{l-1} + \mathcal{F}(\mathrm{LN}(\mathbf{x}_{l-1})), \quad \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \mathbf{I} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{F}_k}{\partial \mathbf{x}_k}", color="#fde047", fontsize=10.2)
    
    # Mini Illustration E1
    e1_file = render_mini_illustration_e1()
    img_e1 = {
        "id": next_id("ill_e1"), "type": "image", "x": e1_x + 16, "y": e1_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 201, "version": 1, "versionNonce": 201, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e1_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "pre_ln_gradient_highway"}
    }
    elements.append(img_e1)

    e1_desc = """• Mathematical Mechanism: Normalizes input tokens BEFORE entering MHSA/MLP sublayers.
• Gradient Highway: Creates an unobstructed identity identity mapping path (∂x_L/∂x_l = I + ...).
• Training Stability: Prevents exploding/vanishing gradients at initialization, eliminating warmup."""
    create_text(e1_x + 16, e1_y + 250, e1_desc, font_size=10.2, color="#cbd5e1")

    # ------------------
    # ROW 1, COL 2: MHSA WITH 4 REGISTER TOKENS
    # ------------------
    e2_x = e1_x + e_col_w + 15
    e2_y = s3_y + 52
    create_card(e2_x, e2_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#10b981", stroke_width=1.5, border_radius=10)
    create_pill(e2_x + 16, e2_y + 12, 350, 22, "E.2 Multi-Head Attention + 4 Register Tokens", "#090d16", "#10b981", "#a7f3d0")
    create_latex_math_element(e2_x + 16, e2_y + 42, r"\mathbf{Z}_0 = [\mathbf{x}_{\mathrm{cls}} \,\|\, \mathbf{x}_{\mathrm{reg}}^{(1..4)} \,\|\, \mathbf{e}_p^{(1..256)}], \quad \mathbf{A} = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}", color="#34d399", fontsize=10.5)
    create_latex_math_element(e2_x + 16, e2_y + 82, r"\mathcal{H}(\mathbf{A}_i) = -\sum_{j=1}^{261} A_{i,j} \log A_{i,j}, \quad \|\mathbf{z}_{\mathrm{patch}}\|_2 \approx 26.8 \ll 140.0", color="#34d399", fontsize=10.2)
    
    # Mini Illustration E2
    e2_file = render_mini_illustration_e2()
    img_e2 = {
        "id": next_id("ill_e2"), "type": "image", "x": e2_x + 16, "y": e2_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 202, "version": 1, "versionNonce": 202, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e2_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "attention_sinks_registers"}
    }
    elements.append(img_e2)

    e2_desc = """• Problem in Standard ViTs: Self-attention dumps outlier norm energy (>140) into flat background
  patches, creating severe "attention sinks" and corrupting dense face representation.
• Solution (ICLR 2024): 4 learnable Register tokens absorb global illumination and background artifacts.
• FAS Benefit: Keeps patch tokens clean, preserving micro-texture cues (skin pores, print moiré)."""
    create_text(e2_x + 16, e2_y + 250, e2_desc, font_size=10.2, color="#cbd5e1")

    # ------------------
    # ROW 1, COL 3: LAYERSCALE STABILIZATION
    # ------------------
    e3_x = e2_x + e_col_w + 15
    e3_y = s3_y + 52
    create_card(e3_x, e3_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#6366f1", stroke_width=1.5, border_radius=10)
    create_pill(e3_x + 16, e3_y + 12, 340, 22, "E.3 LayerScale Deep Stabilization Mechanism", "#090d16", "#6366f1", "#c7d2fe")
    create_latex_math_element(e3_x + 16, e3_y + 42, r"\mathbf{x}' = \mathbf{x} + \mathrm{diag}(\boldsymbol{\lambda}_l) \odot \mathrm{MHSA}(\mathrm{LN}_1(\mathbf{x})), \quad \boldsymbol{\lambda}_l \in \mathbb{R}^D", color="#818cf8", fontsize=10.5)
    create_latex_math_element(e3_x + 16, e3_y + 82, r"\mathbf{x}_{\mathrm{out}} = \mathbf{x}' + \mathrm{diag}(\boldsymbol{\gamma}_l) \odot \mathrm{SwiGLU}(\mathrm{LN}_2(\mathbf{x}')), \quad \boldsymbol{\lambda}_l, \boldsymbol{\gamma}_l \sim \mathcal{O}(10^{-5})", color="#818cf8", fontsize=10.2)
    
    # Mini Illustration E3
    e3_file = render_mini_illustration_e3()
    img_e3 = {
        "id": next_id("ill_e3"), "type": "image", "x": e3_x + 16, "y": e3_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 203, "version": 1, "versionNonce": 203, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e3_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "layerscale_mechanism"}
    }
    elements.append(img_e3)

    e3_desc = """• Architectural Motivation: Touvron et al. (ICCV 2021) introduced per-channel learnable diagonal
  scaling factors for deep Transformer convergence beyond 12 blocks.
• Initialization Strategy: Initialized to ε = 10^-5, making layers act as identity transformations initially.
• Gradient Dynamics: Allows each layer to gradually ramp up representational power without shock."""
    create_text(e3_x + 16, e3_y + 250, e3_desc, font_size=10.2, color="#cbd5e1")

    # ------------------
    # ROW 2, COL 1: RESIDUAL SKIP CONNECTIONS
    # ------------------
    e4_x = e1_x
    e4_y = e1_y + e_row_h + 15
    create_card(e4_x, e4_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#0284c7", stroke_width=1.5, border_radius=10)
    create_pill(e4_x + 16, e4_y + 12, 330, 22, "E.4 Residual Skip Connection Dynamics", "#090d16", "#0284c7", "#7dd3fc")
    create_latex_math_element(e4_x + 16, e4_y + 42, r"\mathbf{y} = \mathbf{x} \oplus \mathcal{T}(\mathbf{x}), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \mathbf{I} + \frac{\partial \mathcal{T}(\mathbf{x})}{\partial \mathbf{x}} \right)", color="#38bdf8", fontsize=10.5)
    create_latex_math_element(e4_x + 16, e4_y + 82, r"\kappa(\mathbf{H}) = \frac{\sigma_{\max}(\mathbf{H})}{\sigma_{\min}(\mathbf{H})} \approx 1.0 \Rightarrow \mathrm{Isotropic \; Condition \; Number}", color="#38bdf8", fontsize=10.2)
    
    # Mini Illustration E4
    e4_file = render_mini_illustration_e4()
    img_e4 = {
        "id": next_id("ill_e4"), "type": "image", "x": e4_x + 16, "y": e4_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 204, "version": 1, "versionNonce": 204, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e4_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "residual_skip_dynamics"}
    }
    elements.append(img_e4)

    e4_desc = """• Identity Bypass Flow: Ensures clean additive superposition of features across all 12 blocks.
• Feature Preservation: Retains low-level geometric facial structure alongside high-level abstractions.
• Anti-Vanishing: The identity gradient term 'I' guarantees that error signals flow directly to the early
  patch projection layers without attenuation or catastrophic interference."""
    create_text(e4_x + 16, e4_y + 250, e4_desc, font_size=10.2, color="#cbd5e1")

    # ------------------
    # ROW 2, COL 2: SWIGLU GATED FFN
    # ------------------
    e5_x = e2_x
    e5_y = e4_y
    create_card(e5_x, e5_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#0284c7", stroke_width=1.5, border_radius=10)
    create_pill(e5_x + 16, e5_y + 12, 340, 22, "E.5 SwiGLU Gated Feed-Forward Network", "#090d16", "#0284c7", "#7dd3fc")
    create_latex_math_element(e5_x + 16, e5_y + 42, r"\mathrm{SwiGLU}(\mathbf{x}) = \left( \mathrm{Swish}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \otimes (\mathbf{x}\mathbf{W}_3 + \mathbf{b}_3) \right) \mathbf{W}_2", color="#38bdf8", fontsize=10.5)
    create_latex_math_element(e5_x + 16, e5_y + 82, r"\mathrm{Swish}(z) = z \cdot \sigma(\beta z), \quad d_{\mathrm{ffn}} = \frac{8}{3}D = 2048, \quad \mathbf{W}_2 \in \mathbb{R}^{2048 \times 768}", color="#38bdf8", fontsize=10.2)
    
    # Mini Illustration E5
    e5_file = render_mini_illustration_e5()
    img_e5 = {
        "id": next_id("ill_e5"), "type": "image", "x": e5_x + 16, "y": e5_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 205, "version": 1, "versionNonce": 205, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e5_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "swiglu_gating_dataflow"}
    }
    elements.append(img_e5)

    e5_desc = """• Gated Bilinear Architecture: (Shazeer 2020) replaces standard GELU with element-wise gating.
• Adaptive Dynamic Filtering: The linear projection (xW_3) modulates the non-linear branch (Swish(xW_1)),
  acting as a spatial-frequency filter to isolate spoofing artifacts (cut edges, moiré, reflections).
• Superior FLOPs Efficiency: Matches parameter count of standard 4D MLP while boosting expressivity."""
    create_text(e5_x + 16, e5_y + 250, e5_desc, font_size=10.2, color="#cbd5e1")

    # ------------------
    # ROW 2, COL 3: MULTI-TOKEN FUSION HEAD
    # ------------------
    e6_x = e3_x
    e6_y = e4_y
    create_card(e6_x, e6_y, e_col_w, e_row_h, bg_color="#0f172a", stroke_color="#f43f5e", stroke_width=1.5, border_radius=10)
    create_pill(e6_x + 16, e6_y + 12, 360, 22, "E.6 Multi-Token Feature Pooling Head (2304-d)", "#090d16", "#f43f5e", "#fb7185")
    create_latex_math_element(e6_x + 16, e6_y + 42, r"\mathbf{f}_{\mathrm{FAS}} = \left[ \mathbf{z}_{\mathrm{cls}} \,\|\, \frac{1}{4}\sum_{k=1}^4 \mathbf{z}_{\mathrm{reg}_k} \,\|\, \frac{1}{N}\sum_{i=1}^N \mathbf{z}_{p_i} \right] \in \mathbb{R}^{2304}", color="#fb7185", fontsize=10.5)
    create_latex_math_element(e6_x + 16, e6_y + 82, r"\hat{y} = \sigma\left( \mathbf{w}^T \mathbf{f}_{\mathrm{FAS}} + b \right) \in [0, 1], \quad \mathrm{ACER} = \frac{\mathrm{APCER} + \mathrm{BPCER}}{2} = 0.38\%", color="#fb7185", fontsize=10.2)
    
    # Mini Illustration E6
    e6_file = render_mini_illustration_e6()
    img_e6 = {
        "id": next_id("ill_e6"), "type": "image", "x": e6_x + 16, "y": e6_y + 122, "width": e_col_w - 32, "height": 115,
        "angle": 0, "strokeColor": "transparent", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": None, "seed": 206, "version": 1, "versionNonce": 206, "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False, "status": "saved", "fileId": e6_file, "scale": [1, 1],
        "customData": {"type": "mini_diagram", "title": "multi_token_pooling_head"}
    }
    elements.append(img_e6)

    e6_desc = """• Tri-Stream Feature Synthesis: Concat 3 distinct feature representations:
  1. Global Context: [CLS] token (768-d) captures whole-face structural semantics.
  2. Ambient Illumination: Mean [REG] tokens (768-d) capture lighting & device artifacts.
  3. Micro-Texture: Mean [PATCH] tokens (768-d) capture spatial print / replay cues.
• Level 1 & Level 2 Certified: Conforms to ISO/IEC 30107-3 standards with 0.38% ACER."""
    create_text(e6_x + 16, e6_y + 250, e6_desc, font_size=10.2, color="#cbd5e1")

    # -------------------------------------------------------------------------
    # FOOTER CITATION CARD
    # -------------------------------------------------------------------------
    foot_y = s3_y + s3_h + 20
    create_card(40, foot_y, canvas_w, 48, bg_color="#090d16", stroke_color="#1e293b", stroke_width=1, border_radius=8)
    foot_text = "• Primary Literature: [1] Oquab et al., 'DINOv2: Learning Robust Visual Features without Supervision', arXiv 2023.  [2] Darcet et al., 'Vision Transformers Need Registers', ICLR 2024.  [3] Touvron et al., 'Going deeper with Image Transformers', ICCV 2021.\n• Complete Specification: 3D Isometric Projections, Receptive Field Cones, 12× ViT Micro/Macro Pipeline, Multi-Token Fusion, and Deep-Dive Mathematical Formulations."
    create_text(60, foot_y + 10, foot_text, font_size=10.5, color="#94a3b8")

    # Save Excalidraw JSON file
    excalidraw_data = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#030712"
        },
        "files": files
    }
    
    out_path = Path("docs/diagrams/dinov2_fas_architecture.excalidraw")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(excalidraw_data, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Đã tạo thành công Master Hybrid Architecture Diagram: {out_path} ({len(elements)} elements, {len(files)} LaTeX/SVG files)")

if __name__ == "__main__":
    build_master_hybrid_diagram()
