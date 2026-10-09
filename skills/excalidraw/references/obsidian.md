# Obsidian Excalidraw Plugin Integration Reference

Reference guide for creating and managing diagrams compatible with **Zsolt Viczián's Obsidian-Excalidraw Plugin** ([github.com/zsviczian/obsidian-excalidraw-plugin](https://github.com/zsviczian/obsidian-excalidraw-plugin)).

---

## 1. File Format (`.excalidraw.md`)

Obsidian Excalidraw files are standard Markdown files that contain both human-readable text and an embedded Excalidraw JSON scene:

```markdown
---
excalidraw-plugin: parsed
tags: [excalidraw, architecture, faceid-jp]
---
# Text Elements
1. Input Face (224×224×3) ^elem_txt-1
2. Patch Projection + 4 Registers ^elem_txt-2
3. Frozen DINOv2 ViT-B/14 (86M) ^elem_txt-3

%%
# Drawing
```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://github.com/zsviczian/obsidian-excalidraw-plugin",
  "elements": [...],
  "appState": {
    "viewBackgroundColor": "#ffffff",
    "gridSize": 20
  },
  "files": {}
}
```
%%
```

---

## 2. Key Advantages in Knowledge Management & SDD

1. **Full-Text Search Indexing**:
   - All diagram text blocks are mirrored in `# Text Elements`. Searching keywords inside Obsidian or VS Code immediately finds and navigates to the diagram.
2. **Bi-directional Wikilinks**:
   - Diagram elements can link to Markdown documents via `[[Document Name]]` or `[Text](file:///...)`.
3. **Seamless Vault Embedding**:
   - Embed interactive diagrams inside any note using standard transclusion:
     ```markdown
     ![[apiface2024_system_architecture.excalidraw]]
     ```
4. **Automated Conversion Utility**:
   - Run `python3 scripts/excalidraw_to_obsidian.py` to batch-convert `.excalidraw` JSON into `.excalidraw.md`.

---

## 3. Supported Diagrams in Workspace

| Diagram Name | Excalidraw JSON | Obsidian Markdown | PNG Render |
|---|---|---|---|
| **System Architecture** | [`docs/diagrams/apiface2024_system_architecture.excalidraw`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/apiface2024_system_architecture.excalidraw) | [`docs/diagrams/apiface2024_system_architecture.excalidraw.md`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/apiface2024_system_architecture.excalidraw.md) | [`PNG`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/apiface2024_system_architecture.png) |
| **DINOv2 FAS Architecture** | [`docs/diagrams/dinov2_fas_architecture.excalidraw`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_fas_architecture.excalidraw) | [`docs/diagrams/dinov2_fas_architecture.excalidraw.md`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_fas_architecture.excalidraw.md) | [`PNG`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_fas_architecture.png) |
| **DINOv2 ViT-B/14 Deep Dive** | [`docs/diagrams/dinov2_vit_b14_architecture_details.excalidraw`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_vit_b14_architecture_details.excalidraw) | [`docs/diagrams/dinov2_vit_b14_architecture_details.excalidraw.md`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_vit_b14_architecture_details.excalidraw.md) | [`PNG`](file:///media/linhnx/769a8e4d-ac22-4040-a57b-fc1526d97b225/workspace/workdir_FaceJP/apiface2024/docs/diagrams/dinov2_vit_b14_architecture_details.png) |
