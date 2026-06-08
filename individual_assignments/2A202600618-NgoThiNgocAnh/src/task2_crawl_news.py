"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://nld.com.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-196250821135822527.htm",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
]

# Cấu hình browser — giả lập trình duyệt thật để qua anti-bot
BROWSER_CONFIG = BrowserConfig(
    browser_type="chromium",
    headless=True,
    user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    headers={
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    },
    verbose=False,
)

# Cấu hình crawl mỗi trang
RUN_CONFIG = CrawlerRunConfig(
    extraction_strategy=NoExtractionStrategy(),
    wait_for="body",
    page_timeout=30_000,          # 30s timeout
    delay_before_return_html=2.5, # chờ JS render
    remove_overlay_elements=True,
    simulate_user=True,           # giả lập scroll, move chuột
)


def extract_title_from_markdown(markdown: str) -> str:
    """Lấy tiêu đề từ dòng markdown đầu tiên bắt đầu bằng '#'."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    # Fallback: dòng không rỗng đầu tiên
    for line in markdown.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "Unknown"


async def crawl_article(crawler: AsyncWebCrawler, url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    result = await crawler.arun(url=url, config=RUN_CONFIG)

    if not result.success:
        # Trả về dict lỗi thay vì raise để không dừng toàn bộ pipeline
        return {
            "url": url,
            "title": "CRAWL_FAILED",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"ERROR: {result.error_message}",
            "success": False,
        }

    markdown = result.markdown or ""

    # Ưu tiên title từ metadata (thẻ <title> / og:title), fallback sang markdown
    title = (
        (result.metadata or {}).get("title")
        or (result.metadata or {}).get("og:title")
        or extract_title_from_markdown(markdown)
        or "Unknown"
    )

    return {
        "url": url,
        "title": title.strip(),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
        "success": True,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    # Dùng một crawler instance duy nhất cho toàn bộ — tiết kiệm tài nguyên
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            try:
                article = await crawl_article(crawler, url)
            except Exception as exc:
                print(f"  ✗ Exception: {exc}")
                article = {
                    "url": url,
                    "title": "CRAWL_EXCEPTION",
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": f"EXCEPTION: {exc}",
                    "success": False,
                }

            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            status = "✓" if article.get("success") else "✗"
            print(f"  {status} Saved: {filepath}  |  title: {article['title'][:60]}")

            # Nghỉ nhỏ giữa các request tránh bị rate-limit
            if i < len(ARTICLE_URLS):
                await asyncio.sleep(1.5)

    print(f"\nDone! {len(ARTICLE_URLS)} files saved to {DATA_DIR}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())