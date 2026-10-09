# Hướng dẫn Kỹ thuật Chuyển đổi & Trích xuất Tài liệu Nghiên cứu Khoa học (Paper2MD Reference)

## 1. Cơ chế Xử lý Nguồn LaTeX từ arXiv

Khi người dùng cung cấp một arXiv ID (ví dụ `1607.06450` hoặc `2309.16588`):
1. `paper2md.py` gửi HTTP GET request đến `https://arxiv.org/e-print/{arxiv_id}` để lấy gói lưu trữ nguồn.
2. Gói nén (`.tar.gz` hoặc `.tar`) được giải nén vào thư mục tạm thời.
3. Quét toàn bộ cây thư mục để tìm:
   - Các file ảnh tĩnh: `.png`, `.jpg`, `.jpeg`, `.svg`.
   - Các file đồ họa vector: `.pdf`, `.eps`.
4. Sử dụng PyMuPDF (`fitz`) để rasterize các file `.pdf`/`.eps` thành ảnh `.png` với độ phân giải cao `250 DPI` (`dpi / 72.0` zoom matrix).
5. Quét tìm file `.tex` chính (chứa `\begin{document}`).
6. Đệ quy phân giải toàn bộ các lệnh `\input{...}` và `\include{...}` để gộp thành một luồng nội dung đầy đủ.
7. Áp dụng bộ lọc biểu thức chính quy (Regex Filters) để chuyển đổi:
   - `\section{...}` $\to$ `## ...`
   - `\subsection{...}` $\to$ `### ...`
   - `\includegraphics{path}` $\to$ `![Caption](figures/matched_fig.png)`
   - `\begin{equation}` $\to$ `$$\n...\n$$`
   - `\cite{...}`, `\ref{...}` $\to$ Giữ nguyên mã trỏ `[...]`.

---

## 2. Cơ chế Phân tích Layout PDF Đa năng (PyMuPDF)

Với bất kỳ file PDF nào ngoài arXiv (Local PDF hoặc PDF URL từ OpenReview / IEEE / CVPR):
1. Mở tài liệu bằng `fitz.open()`.
2. Trích xuất ảnh nhúng: Gọi `page.get_images(full=True)`, bóc tách byte ảnh gốc (`base_image["image"]`).
3. Lọc bỏ các icon/logo rác có kích thước $< 80\times 80\text{ px}$ hoặc dung lượng $< 3\text{ KB}$.
4. Phân tích khối văn bản (`page.get_text("blocks")`), sắp xếp theo thứ tự tọa độ từ trên xuống dưới, từ trái sang phải.
5. Tự động nhận diện cấu trúc tiêu đề dựa trên quy tắc Section Numbering (`1. Introduction`, `2. Methodology`, `3. Experiments`).
6. Ghép ảnh tương ứng với từng trang vào đúng vị trí nội dung trong Markdown.

---

## 3. Khuyến nghị Tích hợp với Obsidian & Excalidraw

- Markdown sinh ra hoàn toàn tương thích với **Obsidian**:
  - Hỗ trợ xem ảnh tương đối `![Caption](figures/fig.png)`.
  - Hỗ trợ MathJax / KaTeX rendering tức thì cho mọi công thức toán học.
- Các biểu đồ hoặc kiến trúc trích xuất từ paper có thể được sử dụng làm hình nền (*Canvas Reference*) trong Excalidraw để chú thích (*annotate*) hoặc làm sạch lại.
