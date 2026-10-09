#!/usr/bin/env python3
"""
Universal Paper-to-Markdown & Figure Extraction Engine (paper2md)
================================================================
Inspired by timf34/arxiv2md, extended to support:
  1. Any arXiv paper (via arXiv e-print LaTeX Source + author figures OR arXiv HTML/ar5iv)
  2. Any arbitrary PDF paper (Local file or Remote URL) with automatic image extraction
  3. Automatic vector-to-PNG figure conversion (PDF/EPS -> High-Res PNG)
  4. Preservation of all mathematical formulas in standard LaTeX format ($...$, $$...$$)

Usage:
  python3 scripts/paper2md.py 1607.06450 -o docs/research/papers/layernorm
  python3 scripts/paper2md.py 1910.07467 -o docs/research/papers/rmsnorm
  python3 scripts/paper2md.py path/to/local_paper.pdf -o docs/research/papers/my_paper
"""

import os
import re
import sys
import json
import shutil
import tarfile
import gzip
import zipfile
import tempfile
import argparse
from pathlib import Path
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup, NavigableString

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_arxiv_id(raw_input: str) -> str:
    """Extract clean arXiv ID from URL or string."""
    raw = raw_input.strip()
    match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', raw)
    if match:
        return match.group(1)
    match_old = re.search(r'([a-z\-]+/\d{7})', raw)
    if match_old:
        return match_old.group(1)
    return raw

def convert_pdf_page_to_png(pdf_path: Path, output_png_path: Path, dpi: int = 250):
    """Render single-page or multi-page PDF figure to PNG using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) > 0:
            page = doc[0]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(output_png_path))
            doc.close()
            return True
    except Exception as e:
        print(f"    [WARN] Failed to convert PDF figure {pdf_path.name} to PNG: {e}")
    return False

def extract_figures_from_pdf(pdf_path: Path, figures_dir: Path) -> list:
    """Extract all embedded images and rendered visual pages from a PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    extracted_images = []
    
    img_idx = 1
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Filter out tiny icons or logos (e.g. < 50px width/height or < 2KB)
            if base_image.get("width", 0) < 80 or base_image.get("height", 0) < 80 or len(image_bytes) < 3000:
                continue
                
            img_filename = f"figure_p{page_num+1}_{img_idx}.{image_ext}"
            img_path = figures_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            extracted_images.append({
                "page": page_num + 1,
                "path": img_path,
                "filename": img_filename,
                "width": base_image.get("width", 0),
                "height": base_image.get("height", 0)
            })
            img_idx += 1
            
    doc.close()
    return extracted_images

def convert_general_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """Convert any arbitrary PDF into structured Markdown with extracted figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    extracted_imgs = extract_figures_from_pdf(pdf_path, figures_dir)
    
    doc = fitz.open(str(pdf_path))
    md_lines = []
    
    # 1. Title and metadata from first page
    doc_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
    md_lines.append(f"# {doc_title}\n")
    md_lines.append(f"> **Source PDF:** `{pdf_path.name}` ({len(doc)} pages, {len(extracted_imgs)} figures extracted)\n")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        
        md_lines.append(f"\n<!-- Page {page_num + 1} -->\n")
        
        # Sort blocks vertically
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            
            # Simple heuristic for headings
            lines = text.split("\n")
            first_line = lines[0].strip()
            
            # Section header heuristics (e.g., "1. Introduction", "Abstract", "2. Related Work")
            if re.match(r'^(?:[0-9]+\.?\s+[A-Z]|Abstract|Introduction|Conclusion|References|Method|Approach|Experiments)', first_line, re.IGNORECASE) and len(first_line) < 80:
                md_lines.append(f"\n## {first_line}\n")
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    md_lines.append(f"{rest}\n")
            elif re.match(r'^(?:[0-9]+\.[0-9]+\.?\s+[A-Z])', first_line) and len(first_line) < 80:
                md_lines.append(f"\n### {first_line}\n")
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    md_lines.append(f"{rest}\n")
            else:
                # Normal paragraph text
                cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                md_lines.append(f"{cleaned_text}\n")
        
        # Insert any figures extracted on this page
        page_imgs = [img for img in extracted_imgs if img["page"] == page_num + 1]
        for pimg in page_imgs:
            md_lines.append(f"\n![Figure from Page {page_num + 1}](figures/{pimg['filename']})\n")

    doc.close()
    
    output_md_path = output_dir / "paper.md"
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"✓ Converted PDF -> Markdown: {output_md_path} ({len(extracted_imgs)} figures)")
    return output_md_path

def convert_arxiv_source_to_markdown(arxiv_id: str, output_dir: Path) -> Path:
    """Download arXiv LaTeX source tarball, extract author figures and parse TeX to Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Downloading arXiv e-print source for {arxiv_id}...")
    e_print_url = f"https://arxiv.org/e-print/{arxiv_id}"
    resp = requests.get(e_print_url, headers=HEADERS, timeout=30)
    
    if resp.status_code != 200:
        print(f"    [WARN] e-print source download returned {resp.status_code}. Fallback to PDF/HTML...")
        return None
        
    temp_dir = Path(tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id}_"))
    tar_path = temp_dir / "source.tar"
    with open(tar_path, "wb") as f:
        f.write(resp.content)
        
    # Unpack tar or gzip
    try:
        if tarfile.is_tarfile(str(tar_path)):
            with tarfile.open(str(tar_path), "r:*") as tar:
                tar.extractall(path=str(temp_dir))
        else:
            # Single gzipped file
            with gzip.open(str(tar_path), "rb") as gz:
                content = gz.read()
            with open(temp_dir / "main.tex", "wb") as f:
                f.write(content)
    except Exception as e:
        print(f"    [WARN] Error unpacking source tar: {e}")
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        return None

    # Find all figures in unpacked directory and copy/convert to output figures_dir
    copied_figures = {}
    for p in temp_dir.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg", ".svg"]:
                target = figures_dir / p.name
                shutil.copy2(str(p), str(target))
                copied_figures[p.name] = target.name
                copied_figures[p.stem] = target.name
            elif suffix == ".pdf":
                # Convert PDF figure to PNG
                target_png = figures_dir / f"{p.stem}.png"
                if convert_pdf_page_to_png(p, target_png):
                    copied_figures[p.name] = target_png.name
                    copied_figures[p.stem] = target_png.name

    # Find main .tex file
    tex_files = list(temp_dir.rglob("*.tex"))
    if not tex_files:
        print("    [WARN] No .tex files found in source archive.")
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        return None
        
    # Heuristic for main .tex: contains \begin{document}
    main_tex = None
    for tf in tex_files:
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "\\begin{document}" in content:
                main_tex = tf
                break
        except Exception:
            pass
    if not main_tex:
        main_tex = tex_files[0]
        
    print(f"    [+] Found main TeX file: {main_tex.name} and {len(copied_figures)} figures.")
    
    # Read and resolve \input / \include
    def resolve_tex_includes(file_path: Path, visited=None) -> str:
        if visited is None:
            visited = set()
        if file_path in visited or not file_path.exists():
            return ""
        visited.add(file_path)
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        def repl_input(match):
            inc_name = match.group(1).strip()
            if not inc_name.endswith(".tex"):
                inc_name += ".tex"
            inc_path = file_path.parent / inc_name
            if not inc_path.exists():
                # Search recursively in temp_dir
                cand = list(temp_dir.rglob(inc_name))
                if cand:
                    inc_path = cand[0]
            if inc_path.exists():
                return resolve_tex_includes(inc_path, visited)
            return ""
            
        text = re.sub(r'\\(?:input|include)\{([^}]+)\}', repl_input, text)
        return text

    full_tex = resolve_tex_includes(main_tex)
    
    # Parse LaTeX to Markdown
    md_content = parse_latex_to_markdown(full_tex, arxiv_id, copied_figures)
    
    out_md = output_dir / "paper.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    shutil.rmtree(str(temp_dir), ignore_errors=True)
    print(f"✓ Converted arXiv LaTeX -> Markdown: {out_md} ({len(copied_figures)} figures saved)")
    return out_md

def parse_latex_to_markdown(tex: str, arxiv_id: str, copied_figures: dict) -> str:
    """Parse raw LaTeX string into clean Markdown with LaTeX math and figure links."""
    # 1. Remove comments
    tex = re.sub(r'(?<!\\)%.*$', '', tex, flags=re.MULTILINE)
    
    # 2. Extract Title
    title_m = re.search(r'\\title\{([^}]+)\}', tex)
    title = title_m.group(1).replace("\\\\", " ").strip() if title_m else f"arXiv Paper {arxiv_id}"
    # Remove TeX formatting from title
    title = re.sub(r'\\[a-zA-Z]+\{?|[\}]', '', title).strip()
    
    # 3. Extract Abstract
    abstract_m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, flags=re.DOTALL)
    abstract = abstract_m.group(1).strip() if abstract_m else ""
    
    # 4. Body after \begin{document}
    body_m = re.search(r'\\begin\{document\}(.*)', tex, flags=re.DOTALL)
    body = body_m.group(1) if body_m else tex
    # Remove \end{document}
    body = re.sub(r'\\end\{document\}.*', '', body, flags=re.DOTALL)
    
    # Replace Sections
    body = re.sub(r'\\section\*?\{([^}]+)\}', r'\n## \1\n', body)
    body = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n### \1\n', body)
    body = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n#### \1\n', body)
    body = re.sub(r'\\paragraph\*?\{([^}]+)\}', r'\n**\1** ', body)
    
    # Replace Figures \includegraphics
    def repl_fig(match):
        fig_target = match.group(1).strip()
        base_name = Path(fig_target).name
        stem_name = Path(fig_target).stem
        
        resolved = copied_figures.get(base_name) or copied_figures.get(stem_name) or copied_figures.get(f"{stem_name}.png")
        if resolved:
            return f"\n\n![Figure: {stem_name}](figures/{resolved})\n\n"
        return f"\n\n*(Figure: {fig_target})*\n\n"
        
    body = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', repl_fig, body)
    
    # Clean Figure and Table environments
    body = re.sub(r'\\caption\{([^}]+)\}', r'\n*Caption: \1*\n', body)
    body = re.sub(r'\\begin\{(?:figure|table)\*?\}(.*?)\\end\{(?:figure|table)\*?\}', r'\n\1\n', body, flags=re.DOTALL)
    
    # Math environment conversion
    # \[ ... \] -> $$ ... $$
    body = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', body, flags=re.DOTALL)
    # \begin{equation} ... \end{equation} -> $$ ... $$
    body = re.sub(r'\\begin\{(?:equation|align|gather|multline)\*?\}(.*?)\\end\{(?:equation|align|gather|multline)\*?\}', r'\n$$\n\1\n$$\n', body, flags=re.DOTALL)
    
    # Clean citations and refs
    body = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', body)
    body = re.sub(r'\\ref\{([^}]+)\}', r'[\1]', body)
    body = re.sub(r'\\label\{([^}]+)\}', '', body)
    
    # Clean font commands \textbf, \textit, \emph
    body = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', body)
    body = re.sub(r'\\(?:textit|emph)\{([^}]+)\}', r'*\1*', body)
    body = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', body)
    
    # Clean multiple spaces and blank lines
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    md_lines = [
        f"# {title}\n",
        f"> **arXiv Identifier:** [`arXiv:{arxiv_id}`](https://arxiv.org/abs/{arxiv_id})\n",
    ]
    if abstract:
        clean_abs = re.sub(r'\\[a-zA-Z]+\{?|[\}]', '', abstract).strip()
        md_lines.append(f"## Abstract\n\n{clean_abs}\n")
        
    md_lines.append(body)
    return "\n".join(md_lines)

def convert_arxiv_html_to_markdown(arxiv_id: str, output_dir: Path) -> Path:
    """Fallback: Convert arXiv HTML5 representation to Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    print(f"[*] Trying arXiv HTML at {html_url}...")
    resp = requests.get(html_url, headers=HEADERS, timeout=25)
    
    if resp.status_code != 200:
        # Try ar5iv
        html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        resp = requests.get(html_url, headers=HEADERS, timeout=25)
        
    if resp.status_code != 200:
        print(f"    [WARN] arXiv HTML returned {resp.status_code}.")
        return None
        
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Download images
    img_idx = 1
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src")
        if src:
            if not src.startswith("http"):
                src = f"https://arxiv.org/html/{arxiv_id}/{src}"
            try:
                img_resp = requests.get(src, headers=HEADERS, timeout=10)
                if img_resp.status_code == 200:
                    ext = Path(src).suffix or ".png"
                    local_img_name = f"arxiv_fig_{img_idx}{ext}"
                    with open(figures_dir / local_img_name, "wb") as f:
                        f.write(img_resp.content)
                    img_tag["src"] = f"figures/{local_img_name}"
                    img_idx += 1
            except Exception:
                pass
                
    # Extract title & text
    title = soup.find("h1", class_="ltx_title")
    title_text = title.get_text().strip() if title else f"arXiv Paper {arxiv_id}"
    
    body_md = f"# {title_text}\n\n> **Source:** [`arXiv:{arxiv_id}`](https://arxiv.org/abs/{arxiv_id})\n\n"
    
    # Process main article sections
    for sec in soup.find_all(["section", "div"], class_=re.compile(r'ltx_section|ltx_para|ltx_figure')):
        body_md += sec.get_text() + "\n\n"
        
    out_md = output_dir / "paper.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(body_md)
        
    print(f"✓ Converted arXiv HTML -> Markdown: {out_md}")
    return out_md

def convert_paper_universal(input_source: str, output_dir: Path) -> Path:
    """Universal Dispatcher: Auto-detects arXiv ID, PDF URL, or Local PDF."""
    input_str = input_source.strip()
    output_dir = Path(output_dir)
    
    # 1. Local PDF File
    if Path(input_str).is_file() and input_str.lower().endswith(".pdf"):
        print(f"[+] Processing local PDF: {input_str}")
        return convert_general_pdf_to_markdown(Path(input_str), output_dir)
        
    # 2. Remote PDF URL (not arXiv)
    if input_str.startswith("http") and input_str.lower().endswith(".pdf") and "arxiv.org" not in input_str:
        print(f"[+] Downloading remote PDF: {input_str}")
        resp = requests.get(input_str, headers=HEADERS, timeout=30)
        temp_pdf = Path(tempfile.gettempdir()) / "downloaded_paper.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(resp.content)
        return convert_general_pdf_to_markdown(temp_pdf, output_dir)
        
    # 3. arXiv ID or URL
    arxiv_id = clean_arxiv_id(input_str)
    if re.match(r'^\d{4}\.\d{4,5}(?:v\d+)?$', arxiv_id) or re.match(r'^[a-z\-]+/\d{7}$', arxiv_id):
        print(f"[+] Detected arXiv Paper ID: {arxiv_id}")
        
        # Strategy A: LaTeX Source with original author figures (Best quality)
        res = convert_arxiv_source_to_markdown(arxiv_id, output_dir)
        if res and res.exists():
            return res
            
        # Strategy B: arXiv HTML / ar5iv
        res = convert_arxiv_html_to_markdown(arxiv_id, output_dir)
        if res and res.exists():
            return res
            
        # Strategy C: Download arXiv PDF and parse with PyMuPDF
        print(f"[*] Downloading arXiv PDF for {arxiv_id}...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            temp_pdf = Path(tempfile.gettempdir()) / f"{arxiv_id}.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(resp.content)
            return convert_general_pdf_to_markdown(temp_pdf, output_dir)
            
    print(f"[-] Could not process input source: {input_source}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Universal Paper-to-Markdown & Figure Extraction (paper2md)")
    parser.add_argument("input", type=str, help="arXiv ID, arXiv URL, PDF URL, or local PDF file path")
    parser.add_argument("--output", "-o", type=str, default="docs/research/converted_paper", help="Output directory")
    
    args = parser.parse_args()
    res = convert_paper_universal(args.input, Path(args.output))
    if res:
        print(f"\n✨ Successfully converted paper to Markdown:\n   -> {res.resolve()}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
