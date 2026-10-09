---
name: paper2md
description: "Universal Paper-to-Markdown & Figure Extraction Engine — Chuyển đổi bất kỳ bài báo khoa học (arXiv ID, arXiv URL, Local PDF, Remote PDF) thành tài liệu Markdown chuẩn GitHub với đầy đủ công thức LaTeX ($...$, $$...$$) và tự động bóc tách 100% hình ảnh/đồ thị gốc của tác giả (tự động chuyển đổi vector PDF/EPS sang PNG độ phân giải cao)."
---

# Paper2MD — Universal Paper Ingestion & Figure Extraction Skill

Skill chuyên biệt giúp AI Agent tải, giải nén và chuyển đổi bất kỳ bài báo nghiên cứu khoa học nào từ arXiv hoặc tệp PDF bất kỳ thành định dạng **Markdown chuẩn hóa**, bảo toàn trọn vẹn:
1. **Công thức Toán học LaTeX:** Chuyển đổi chính xác sang `$..$` (inline) và `$$..$$` (block equations).
2. **Hình ảnh & Đồ thị Gốc:** Tự động trích xuất toàn bộ ảnh bitmap và vector (`.pdf`, `.eps`) từ gói nguồn LaTeX của tác giả hoặc bóc tách từ các trang PDF, tự động convert sang `.png` độ nét cao lưu vào thư mục `figures/`.
3. **Cấu trúc Tiêu đề & Bảng biểu:** Tự động nhận diện H1, H2, H3, Abstract, Caption, Citations và Tables.

---

## 🚀 Lệnh Thực thi Nhanh (CLI Quickstart)

```bash
# 1. Chuyển đổi từ arXiv ID bất kỳ
python3 .agents/skills/paper2md/scripts/paper2md.py 1607.06450 -o docs/research/converted_papers/layernorm

# 2. Chuyển đổi từ file PDF cục bộ
python3 .agents/skills/paper2md/scripts/paper2md.py /path/to/my_paper.pdf -o docs/research/converted_papers/my_paper

# 3. Chuyển đổi từ link URL PDF từ xa (OpenReview, CVPR, NeurIPS)
python3 .agents/skills/paper2md/scripts/paper2md.py https://openreview.net/pdf?id=xxx -o docs/research/converted_papers/openreview_paper

# 4. Chạy Batch Survey hàng loạt paper
python3 .agents/skills/paper2md/scripts/batch_convert_papers.py
```

---

## 🛠️ Kiến trúc Xử lý Đa tầng (Multi-Strategy Pipeline)

```
                              INPUT (arXiv ID / PDF URL / Local PDF)
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
        [Đầu vào là arXiv ID]                                         [Đầu vào là File PDF]
                 │                                                             │
        ┌────────┴────────┐                                                    │
        ▼                 ▼                                                    │
┌───────────────┐ ┌───────────────┐                                            │
│ Chiến lược A  │ │ Chiến lược B  │                                            │
│ arXiv e-print │ │  arXiv HTML   │                                            │
│ LaTeX Source  │ │    (ar5iv)    │                                            ▼
└───────┬───────┘ └───────┬───────┘                                   ┌─────────────────┐
        │                 │                                           │  Chiến lược C   │
        ▼                 ▼                                           │ PyMuPDF (fitz)  │
┌─────────────────────────────────┐                                   │ Layout Analysis │
│  Trích xuất .tex & Convert ảnh  │                                   │ Image Extraction│
│   Vector PDF/EPS -> PNG (250dpi)│                                   └────────┬────────┘
└───────────────┬─────────────────┘                                            │
                │                                                              │
                └──────────────────────────────┬───────────────────────────────┘
                                               ▼
                              ┌─────────────────────────────────┐
                              │    OUTPUT MARKDOWN DIRECTORY    │
                              │    • paper.md                   │
                              │    • figures/                   │
                              │      ├── fig1.png               │
                              │      └── fig2.png               │
                              └─────────────────────────────────┘
```

---

## 📦 Cấu trúc Thư mục

```
.agents/skills/paper2md/
├── SKILL.md
├── scripts/
│   ├── paper2md.py              # Engine chuyển đổi universal
│   └── batch_convert_papers.py  # Script chạy batch ingestion hàng loạt paper
└── references/
    └── universal_paper_ingestion.md
```

---

## 💡 Khi nào nên Sử dụng Skill Này?

1. **Khi cần nghiên cứu sâu một bài báo:** Thay vì đọc tóm tắt sơ sài, hãy ingest toàn bộ bài báo vào Markdown để nắm rõ từng công thức, siêu tham số thực nghiệm và điều kiện đo lường.
2. **Khi cần lấy hình ảnh & đồ thị thực nghiệm gốc:** Tự động trích xuất các biểu đồ benchmark, ablation study, loss curves trực tiếp từ tác giả mà không cần vẽ lại thủ công.
3. **Khi cần làm Literature Review / State-of-the-art Survey:** Chạy batch converter để nạp hàng chục paper vào kho tri thức phục vụ System One (Jevify) & System Two (Reasoning).
