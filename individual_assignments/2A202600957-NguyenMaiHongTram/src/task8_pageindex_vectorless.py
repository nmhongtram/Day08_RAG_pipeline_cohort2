"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document (cây mục lục / heading) thay vì
embedding + ANN search. Dùng làm fallback khi hybrid search (Task 9)
không tìm được kết quả đủ tốt.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key, set PAGEINDEX_API_KEY trong .env
    3. Upload documents (PDF gốc trong data/landing/legal/)
    4. Query sử dụng PageIndex API (submit_query → poll → get_retrieval)
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Cache mapping doc_id -> tên file gốc, để khỏi phải re-upload mỗi lần search
DOC_MAP_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_map.json"


def _load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        return json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_map(doc_map: dict) -> None:
    DOC_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> dict:
    """
    Upload toàn bộ văn bản pháp luật (PDF gốc trong data/landing/legal/)
    lên PageIndex — PageIndex xử lý trực tiếp PDF (tự build tree + OCR),
    nên ta dùng file gốc thay vì bản markdown đã chuẩn hoá.

    Returns:
        dict mapping {doc_id: filename}
    """
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_map = _load_doc_map()
    for pdf_file in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        if pdf_file.name in doc_map.values():
            continue
        result = client.submit_document(str(pdf_file))
        doc_id = result.get("doc_id")
        if doc_id:
            doc_map[doc_id] = pdf_file.name
            print(f"  ✓ Uploaded: {pdf_file.name} -> doc_id={doc_id}")

    _save_doc_map(doc_map)
    return doc_map


def _wait_until_ready(client, doc_id: str, timeout: float = 60.0, interval: float = 3.0) -> bool:
    """Poll PageIndex cho tới khi document sẵn sàng cho retrieval (hoặc timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.is_retrieval_ready(doc_id):
            return True
        time.sleep(interval)
    return False


def _extract_results(retrieval_result: dict) -> list[dict]:
    """Chuẩn hoá kết quả trả về từ get_retrieval() — API có thể đặt tên field khác nhau."""
    raw_items = (
        retrieval_result.get("results")
        or retrieval_result.get("nodes")
        or retrieval_result.get("data")
        or []
    )
    items = []
    for item in raw_items:
        content = item.get("content") or item.get("text") or item.get("summary") or ""
        score = item.get("score", item.get("relevance_score", 0.0))
        items.append({"content": content, "score": float(score), "raw": item})
    return items


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_map = _load_doc_map()
    if not doc_map:
        doc_map = upload_documents()

    all_results: list[dict] = []
    for doc_id, filename in doc_map.items():
        if not _wait_until_ready(client, doc_id):
            continue

        submission = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submission.get("retrieval_id")
        if not retrieval_id:
            continue

        retrieval = client.get_retrieval(retrieval_id)
        for item in _extract_results(retrieval):
            all_results.append({
                "content": item["content"],
                "score": item["score"],
                "metadata": {"source": filename, "type": "legal", "doc_id": doc_id},
                "source": "pageindex",
            })

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
