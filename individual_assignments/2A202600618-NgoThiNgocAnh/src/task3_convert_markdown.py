"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown[pdf]   # hỗ trợ PDF

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {legal_dir}")
        return

    md = MarkItDown()
    files = [f for f in legal_dir.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".doc")]

    if not files:
        print(f"  ⚠ Không tìm thấy file PDF/DOCX trong {legal_dir}")
        return

    for filepath in files:
        print(f"  Converting: {filepath.name}")
        try:
            if filepath.suffix.lower() == ".doc":
                if not HAS_DOCX2TXT:
                    print(f"  ✗ Bỏ qua {filepath.name}: cần cài 'pip install docx2txt'")
                    continue
                text = docx2txt.process(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(text or "", encoding="utf-8")
            else:
                result = md.convert(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(result.text_content, encoding="utf-8")

            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Lỗi khi convert {filepath.name}: {e}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {news_dir}")
        return

    files = [f for f in news_dir.iterdir() if f.suffix.lower() == ".json"]

    if not files:
        print(f"  ⚠ Không tìm thấy file JSON trong {news_dir}")
        return

    for filepath in sorted(files):
        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Lỗi khi convert {filepath.name}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()