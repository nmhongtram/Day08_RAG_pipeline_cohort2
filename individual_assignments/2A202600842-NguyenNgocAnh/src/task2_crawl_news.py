"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI để crawl nội dung và chuyển sang Markdown.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   (cài Playwright browsers)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# 5 bài báo về nghệ sĩ Việt liên quan tới ma tuý
ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-pha-huy-nao-bo-nguoi-tre-ra-sao-2518285.html",
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo",
    "https://tienphong.vn/ma-tuy-hao-quang-sa-chan-chon-bun-lay-post1763929.tpo",
    "https://thanhnien.vn/phat-ngon-cua-nsut-hanh-thuy-giua-loat-nghe-si-vuong-ma-tuy-185260520162827386.htm",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo bằng Crawl4AI và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

    # Cấu hình browser: headless, không cache để luôn lấy nội dung mới
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        word_count_threshold=10,   # bỏ qua các block quá ngắn (nav, footer...)
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        remove_overlay_elements=True,
        page_timeout=30000,        # timeout 30s mỗi trang
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title") or result.metadata.get("og:title") or "Unknown"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS và lưu JSON."""
    setup_directory()

    success = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"\n[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            content_len = len(article["content_markdown"])
            print(f"  ✓ Saved: {filepath.name}  |  title: {article['title'][:60]}  |  {content_len} chars")
            success += 1
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")

    print(f"\n=== Hoàn thành: {success}/{len(ARTICLE_URLS)} bài báo đã được lưu vào {DATA_DIR} ===")


if __name__ == "__main__":
    asyncio.run(crawl_all())
