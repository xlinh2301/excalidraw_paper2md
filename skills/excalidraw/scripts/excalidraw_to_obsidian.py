#!/usr/bin/env python3
"""
Convert standard Excalidraw JSON (.excalidraw or .json) into Obsidian Excalidraw Markdown (.excalidraw.md)
Format specification based on Zsolt Viczián's obsidian-excalidraw-plugin (https://github.com/zsviczian/obsidian-excalidraw-plugin).
Features:
- Valid YAML frontmatter: excalidraw-plugin: parsed, tags: [excalidraw, architecture]
- Extracted # Text Elements section for full-text search indexing & wikilinks
- Embedded %% # Drawing ```json ... ``` %% section
"""

import json
import sys
from pathlib import Path

def convert_to_obsidian_markdown(excalidraw_path: Path, output_md_path: Path, tags=None):
    if tags is None:
        tags = ["excalidraw", "architecture", "faceid-jp"]
        
    with open(excalidraw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    elements = data.get("elements", [])
    
    # Extract text elements for human readability and markdown searchability
    text_lines = []
    for elem in elements:
        if elem.get("type") == "text" and elem.get("text"):
            txt = elem.get("text").strip()
            elem_id = elem.get("id", "")
            text_lines.append(f"{txt} ^elem_{elem_id}\n")
            
    # Prepare Obsidian Markdown structure
    tags_str = ", ".join(tags)
    frontmatter = f"""---
excalidraw-plugin: parsed
tags: [{tags_str}]
---
"""
    
    body = f"""# Text Elements
{"".join(text_lines)}
%%
# Drawing
```json
{json.dumps(data, indent=2, ensure_ascii=False)}
```
%%
"""
    
    full_content = frontmatter + body
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(full_content)
        
    print(f"✓ Đã xuất bản file Obsidian Excalidraw: {output_md_path} ({len(text_lines)} text blocks)")

def main():
    diagram_dir = Path("docs/diagrams")
    for excal_file in diagram_dir.glob("*.excalidraw"):
        out_md = excal_file.with_suffix(".excalidraw.md")
        convert_to_obsidian_markdown(excal_file, out_md)

if __name__ == "__main__":
    main()
