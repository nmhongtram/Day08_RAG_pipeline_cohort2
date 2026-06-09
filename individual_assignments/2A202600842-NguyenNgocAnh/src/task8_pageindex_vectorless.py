"""
Task 8 — PageIndex Vectorless RAG.

Sử dụng PageIndex Cloud API (pageindex.ai) để tạo vectorless RAG pipeline.

=============================================================================
PAGEINDEX LÀ GÌ?

PageIndex thực hiện RAG mà KHÔNG dùng vector DB hay chunking.
Thay vào đó, nó:
    1. Build hierarchical tree index (giống "table of contents") từ document
    2. Dùng LLM reasoning để tìm nodes liên quan trong cây — thay vì
       cosine similarity search

So sánh với pipeline vector-based (Task 4–7):
    Vector RAG:  embed → cosine_similarity → top-k chunks
    PageIndex:   tree_index → LLM_reasoning → relevant_sections

Ưu điểm:
    - Không mất context do chunking thủ công
    - Retrieval explainable (có reasoning chain)
    - Tốt hơn cho document dài có cấu trúc phức tạp (pháp luật, báo cáo)

Nhược điểm:
    - Cần API call LLM mỗi lần query (chậm hơn, có chi phí)
    - Cần upload document lên cloud

Dùng làm FALLBACK khi hybrid search (Task 5+6) không có kết quả tốt.

=============================================================================

Cài đặt:
    pip install pageindex

Đăng ký API key tại: https://dash.pageindex.ai/api-keys
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env từ root project (không phụ thuộc vào cwd)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR  = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR       = Path(__file__).parent.parent / "data" / "landing"
CACHE_FILE        = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# Cache client và doc_ids
_client   = None
_doc_ids: dict[str, str] = {}   # filename → doc_id


# =============================================================================
# Helpers
# =============================================================================

def _get_client():
    """Khởi tạo và cache PageIndexClient."""
    global _client
    if _client is None:
        if not PAGEINDEX_API_KEY:
            raise EnvironmentError(
                "PAGEINDEX_API_KEY chưa được set.\n"
                "Đăng ký tại: https://dash.pageindex.ai/api-keys\n"
                "Sau đó thêm vào file .env: PAGEINDEX_API_KEY=pi_xxx"
            )
        from pageindex import PageIndexClient
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def _load_doc_id_cache() -> dict:
    """Load cache doc_id đã upload từ file JSON."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_doc_id_cache(cache: dict):
    """Lưu cache doc_id xuống file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Upload Documents
# =============================================================================

def upload_documents(force_reupload: bool = False) -> dict[str, str]:
    """
    Upload PDF documents lên PageIndex Cloud.

    PageIndex API chỉ hỗ trợ PDF — upload từ data/landing/legal/*.pdf.
    Dùng cache để tránh re-upload file đã có.
    PageIndex sẽ build hierarchical tree index từ mỗi document.

    Args:
        force_reupload: Nếu True, upload lại tất cả kể cả đã có cache.

    Returns:
        Dict {filename: doc_id}
    """
    client = _get_client()
    cache  = {} if force_reupload else _load_doc_id_cache()

    # PageIndex API chỉ hỗ trợ PDF — dùng file gốc từ landing/legal
    pdf_files = sorted((LANDING_DIR / "legal").glob("*.pdf"))
    if not pdf_files:
        print("⚠ Không tìm thấy file PDF trong data/landing/legal/")
        return cache

    print(f"Uploading {len(pdf_files)} PDF documents lên PageIndex ...")
    for pdf_file in pdf_files:
        key = pdf_file.name
        if key in cache and not force_reupload:
            print(f"  ✓ Skip (cached): {key}  →  {cache[key]}")
            continue

        print(f"  Uploading: {key} ...", end=" ")
        try:
            result = client.submit_document(str(pdf_file))
            doc_id = result["doc_id"]
            cache[key] = doc_id
            print(f"✓  doc_id={doc_id}")
        except Exception as e:
            print(f"✗  Lỗi: {e}")

    _save_doc_id_cache(cache)
    print(f"\n✓ Uploaded. Cache lưu tại: {CACHE_FILE}")
    return cache


def wait_for_ready(doc_ids: dict[str, str], timeout: int = 120) -> dict[str, str]:
    """
    Đợi cho đến khi tất cả documents sẵn sàng để query.

    Args:
        doc_ids:  Dict {filename: doc_id}
        timeout:  Số giây tối đa chờ

    Returns:
        Dict {filename: doc_id} chỉ gồm những doc đã ready
    """
    client  = _get_client()
    ready   = {}
    pending = dict(doc_ids)
    start   = time.time()

    print(f"Đợi PageIndex processing ({len(pending)} documents) ...")
    while pending and (time.time() - start) < timeout:
        for name, doc_id in list(pending.items()):
            try:
                if client.is_retrieval_ready(doc_id):
                    ready[name] = doc_id
                    del pending[name]
                    print(f"  ✓ Ready: {name}")
            except Exception:
                pass
        if pending:
            time.sleep(3)

    if pending:
        print(f"  ⚠ Timeout ({timeout}s). {len(pending)} documents chưa ready: {list(pending.keys())}")
    return ready


# =============================================================================
# PageIndex Search — hàm chính
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex Cloud API.
    Dùng làm fallback khi hybrid search (Task 5+6) không có kết quả tốt.

    Cơ chế:
        1. Với mỗi document đã upload, gọi submit_query() → retrieval_id
        2. Poll get_retrieval() cho đến khi có kết quả
        3. Gộp kết quả từ tất cả documents, lấy top_k

    Args:
        query:  Câu truy vấn
        top_k:  Số lượng kết quả tối đa

    Returns:
        List of {
            'content' : str,          # Nội dung section được retrieve
            'score'   : float,        # Rank-based score (1.0 = best)
            'metadata': dict,         # source, doc_id, node_title
            'retrieval_source': str   # 'pageindex' — đánh dấu nguồn
        }
        Sorted by score descending.
    """
    client = _get_client()

    # Load doc_ids từ cache
    doc_ids = _load_doc_id_cache()
    if not doc_ids:
        raise RuntimeError(
            "Chưa upload documents. Hãy chạy upload_documents() trước.\n"
            "Hoặc chạy: python src/task8_pageindex_vectorless.py --upload"
        )

    all_results = []

    for filename, doc_id in doc_ids.items():
        try:
            # Gửi query
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            # Poll kết quả (timeout 30s)
            result_data = None
            for _ in range(10):
                ret = client.get_retrieval(retrieval_id)
                status = ret.get("status", "")
                if status == "completed" or ret.get("result"):
                    result_data = ret
                    break
                time.sleep(3)

            if not result_data:
                continue

            # Parse kết quả
            retrieved = result_data.get("result", [])
            if isinstance(retrieved, str):
                # Một số phiên bản trả về string
                all_results.append({
                    "content" : retrieved,
                    "score"   : 1.0,
                    "metadata": {"source": filename, "doc_id": doc_id},
                    "retrieval_source": "pageindex",
                })
            elif isinstance(retrieved, list):
                for rank, item in enumerate(retrieved):
                    text = (
                        item.get("text") or
                        item.get("content") or
                        item.get("summary") or
                        str(item)
                    )
                    score = 1.0 / (rank + 1)   # rank-based score
                    all_results.append({
                        "content" : text,
                        "score"   : round(score, 6),
                        "metadata": {
                            "source"    : filename,
                            "doc_id"    : doc_id,
                            "node_title": item.get("title", ""),
                            "node_id"   : item.get("node_id", ""),
                        },
                        "retrieval_source": "pageindex",
                    })
            elif isinstance(retrieved, dict):
                text = (
                    retrieved.get("text") or
                    retrieved.get("content") or
                    str(retrieved)
                )
                all_results.append({
                    "content" : text,
                    "score"   : 1.0,
                    "metadata": {"source": filename, "doc_id": doc_id},
                    "retrieval_source": "pageindex",
                })

        except Exception as e:
            print(f"  ⚠ Lỗi query doc '{filename}': {e}")
            continue

    # Sort by score descending và lấy top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Task 8 — PageIndex Vectorless RAG")
    parser.add_argument("--upload",  action="store_true", help="Upload documents lên PageIndex")
    parser.add_argument("--force",   action="store_true", help="Force re-upload (bỏ cache)")
    parser.add_argument("--query",   type=str,            help="Query để test search")
    parser.add_argument("--list",    action="store_true", help="Liệt kê documents đã upload")
    args = parser.parse_args()

    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được set trong .env")
        print("  Đăng ký tại: https://dash.pageindex.ai/api-keys")
        sys.exit(1)

    if args.list:
        client   = _get_client()
        docs     = client.list_documents(limit=50)
        doc_list = docs.get("documents") or docs.get("results") or []
        print(f"Documents trên PageIndex ({len(doc_list)}):")
        for d in doc_list:
            print(f"  {d.get('doc_id')} | {d.get('name','?')} | ready={d.get('is_retrieval_ready','?')}")

    elif args.upload:
        doc_ids = upload_documents(force_reupload=args.force)
        ready   = wait_for_ready(doc_ids)
        print(f"\n✓ {len(ready)}/{len(doc_ids)} documents ready to query")

    else:
        query = args.query or "hình phạt sử dụng ma tuý"
        print(f"Query: {query}\n")
        try:
            results = pageindex_search(query, top_k=5)
            if not results:
                print("Không có kết quả.")
            for i, r in enumerate(results, 1):
                src   = r["metadata"].get("source", "?")
                title = r["metadata"].get("node_title", "")
                print(f"  [{i}] score={r['score']:.4f}  {src}  {title}")
                print(f"       {r['content'][:150].strip()}...")
        except Exception as e:
            print(f"Lỗi: {e}")
