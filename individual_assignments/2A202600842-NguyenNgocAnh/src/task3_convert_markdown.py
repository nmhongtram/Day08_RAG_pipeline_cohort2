"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Chiến lược:
    - legal/  : dùng MarkItDown để convert PDF/DOCX → .md
    - news/   : đọc JSON crawled (đã có content_markdown), thêm header metadata → .md
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """
    Convert PDF/DOCX files trong data/landing/legal/ sang Markdown.
    Dùng MarkItDown để extract text từ từng file pháp luật.
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    success = 0
    errors = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"  Converting: {filepath.name} ...", end=" ")
        try:
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm header nguồn gốc để dễ trace
            header = f"---\nsource_file: {filepath.name}\nsource_type: legal_document\n---\n\n"
            output_path.write_text(header + result.text_content, encoding="utf-8")

            size_kb = output_path.stat().st_size // 1024
            print(f"✓  →  {output_path.name}  ({size_kb} KB)")
            success += 1
        except Exception as e:
            print(f"✗  Lỗi: {e}")
            errors += 1

    print(f"  Kết quả legal: {success} thành công, {errors} lỗi")
    return success


def convert_news_articles():
    """
    Convert JSON crawled articles trong data/landing/news/ sang Markdown.
    JSON đã chứa content_markdown từ Task 2, chỉ cần thêm header metadata.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"  Converting: {filepath.name} ...", end=" ")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title       = data.get("title", "Unknown")
            url         = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            content     = data.get("content_markdown", "")

            # Header YAML frontmatter + nội dung
            header = (
                f"---\n"
                f"title: \"{title}\"\n"
                f"source_url: {url}\n"
                f"date_crawled: {date_crawled}\n"
                f"source_type: news_article\n"
                f"---\n\n"
                f"# {title}\n\n"
                f"**Nguồn:** [{url}]({url})  \n"
                f"**Ngày crawl:** {date_crawled}\n\n"
                f"---\n\n"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + content, encoding="utf-8")

            size_kb = max(1, output_path.stat().st_size // 1024)
            print(f"✓  →  {output_path.name}  ({size_kb} KB)")
            success += 1
        except Exception as e:
            print(f"✗  Lỗi: {e}")
            errors += 1

    print(f"  Kết quả news: {success} thành công, {errors} lỗi")
    return success


def convert_all():
    """Convert toàn bộ files từ data/landing/ → data/standardized/."""
    print("=" * 60)
    print("Task 3: Convert sang Markdown (MarkItDown)")
    print("=" * 60)

    print("\n--- Văn bản pháp luật (legal/) ---")
    legal_count = convert_legal_docs()

    print("\n--- Bài báo (news/) ---")
    news_count = convert_news_articles()

    total = legal_count + news_count
    print(f"\n{'=' * 60}")
    print(f"✓ Hoàn thành! Tổng: {total} files")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"    legal/: {legal_count} files")
    print(f"    news/ : {news_count} files")
    print("=" * 60)


if __name__ == "__main__":
    convert_all()
