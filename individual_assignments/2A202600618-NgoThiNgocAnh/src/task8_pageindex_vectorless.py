"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là vectorless RAG — dùng hierarchical tree index + LLM reasoning
thay vì vector similarity. Không cần vector store, không cần chunking thủ công.

Cài đặt:
    pip install pageindex

Đăng ký API key tại: https://dash.pageindex.ai/api-keys

Lưu ý quan trọng:
    - PageIndex Cloud API chỉ nhận PDF (không nhận .md trực tiếp)
    - upload_documents() tự động convert .md → PDF trước khi upload
    - pageindex_search() dùng chat_completions() + tree index để trả về
      kết quả dạng {content, score, metadata, source} tương thích với
      interface của pipeline hiện tại
    - Cần thêm vào .env: PAGEINDEX_API_KEY=pi-...
                         OPENAI_API_KEY=sk-... (dùng để convert .md → PDF)

Environment (.env):
    PAGEINDEX_API_KEY=pi-...
    OPENAI_API_KEY=sk-...   # Tùy chọn — dùng nếu cần fallback summary

Usage:
    python group_project/retrieval/task8_pageindex.py
"""

from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_ID_CACHE_PATH = Path(__file__).parent / ".pageindex_doc_ids.json"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# Timeout chờ document processing (giây)
UPLOAD_POLL_INTERVAL = 5
UPLOAD_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Helper — .md → PDF conversion (dùng reportlab, không cần OpenAI)
# ---------------------------------------------------------------------------

def _md_to_pdf(md_path: Path, output_dir: Path) -> Path:
    """
    Convert một file .md sang PDF đơn giản dùng reportlab.

    PageIndex Cloud API chỉ nhận PDF — bước convert này là bắt buộc.
    Cài đặt: pip install reportlab

    Args:
        md_path:    Path đến file .md
        output_dir: Thư mục lưu PDF tạm

    Returns:
        Path đến file PDF vừa tạo
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        raise ImportError(
            "Cần cài reportlab để convert .md → PDF:\n"
            "    pip install reportlab"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / (md_path.stem + ".pdf")

    if pdf_path.exists():
        return pdf_path  # Dùng cache nếu đã convert

    text = md_path.read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        # Escape XML characters cho reportlab
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if line.startswith("# "):
            story.append(Paragraph(safe[2:], styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(safe[3:], styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(safe[4:], styles["Heading3"]))
        else:
            story.append(Paragraph(safe, styles["Normal"]))

    doc.build(story)
    return pdf_path


# ---------------------------------------------------------------------------
# Doc ID cache — tránh upload lại những file đã upload
# ---------------------------------------------------------------------------

def _load_doc_id_cache() -> dict[str, str]:
    """Load mapping {filename → doc_id} từ file cache local."""
    if DOC_ID_CACHE_PATH.exists():
        return json.loads(DOC_ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_id_cache(cache: dict[str, str]) -> None:
    """Lưu mapping {filename → doc_id} ra file cache local."""
    DOC_ID_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Quy trình:
        1. Convert từng .md → PDF tạm (vì PageIndex Cloud chỉ nhận PDF)
        2. Submit PDF lên PageIndex để build tree index
        3. Poll cho đến khi processing hoàn tất
        4. Lưu {filename → doc_id} vào cache local

    Returns:
        dict mapping {md_filename → pageindex_doc_id}
    """
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError(
            "Chưa set PAGEINDEX_API_KEY.\n"
            "  1. Đăng ký tại https://pageindex.ai/\n"
            "  2. Lấy API key tại https://dash.pageindex.ai/api-keys\n"
            "  3. Thêm PAGEINDEX_API_KEY=pi-... vào file .env"
        )

    from pageindex import PageIndexClient

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_id_cache()
    pdf_tmp_dir = Path(__file__).parent / ".pdf_cache"

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"  [WARNING] Không tìm thấy .md files trong {STANDARDIZED_DIR}", file=sys.stderr)
        return cache

    newly_uploaded: list[tuple[str, str]] = []  # [(filename, doc_id)]

    for md_file in md_files:
        if md_file.name.startswith("."):
            continue

        filename = md_file.name
        if filename in cache:
            print(f"  ↩ Đã upload trước đó: {filename} → {cache[filename]}")
            continue

        print(f"  ↑ Converting & uploading: {filename}...")
        try:
            pdf_path = _md_to_pdf(md_file, pdf_tmp_dir)
            result = pi.submit_document(str(pdf_path))
            doc_id = result["doc_id"]
            cache[filename] = doc_id
            newly_uploaded.append((filename, doc_id))
            print(f"    ✓ Submitted: {filename} → {doc_id}")
        except Exception as e:
            print(f"    ✗ Lỗi khi upload {filename}: {e}", file=sys.stderr)

    # Lưu cache ngay sau khi submit (trước khi chờ processing)
    _save_doc_id_cache(cache)

    # Poll cho đến khi tất cả doc vừa upload hoàn thành processing
    if newly_uploaded:
        print(f"\n  Chờ processing {len(newly_uploaded)} document(s)...")
        _wait_for_processing(pi, [doc_id for _, doc_id in newly_uploaded])

    return cache


def _wait_for_processing(pi, doc_ids: list[str]) -> None:
    """Poll PageIndex cho đến khi tất cả doc_ids có status = 'completed'."""
    pending = set(doc_ids)
    elapsed = 0

    while pending and elapsed < UPLOAD_TIMEOUT:
        time.sleep(UPLOAD_POLL_INTERVAL)
        elapsed += UPLOAD_POLL_INTERVAL

        completed_this_round = set()
        for doc_id in list(pending):
            try:
                status_info = pi.get_document(doc_id)
                status = status_info.get("status", "unknown")
                if status == "completed":
                    print(f"    ✓ Processing xong: {doc_id}")
                    completed_this_round.add(doc_id)
                elif status == "failed":
                    print(f"    ✗ Processing thất bại: {doc_id}", file=sys.stderr)
                    completed_this_round.add(doc_id)
            except Exception as e:
                print(f"    [WARN] Không lấy được status của {doc_id}: {e}", file=sys.stderr)

        pending -= completed_this_round

    if pending:
        print(
            f"  [WARN] Timeout sau {UPLOAD_TIMEOUT}s — {len(pending)} doc(s) chưa xong: {pending}",
            file=sys.stderr,
        )


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Cách hoạt động:
        1. Gọi chat_completions() với query → PageIndex tự reasoning qua tree index
        2. Dùng get_tree() để lấy cấu trúc → map câu trả lời về các node liên quan
        3. Trả về list results dạng chuẩn tương thích với SimpleRAGPipeline

    Args:
        query:  Câu truy vấn tiếng Việt
        top_k:  Số kết quả tối đa trả về

    Returns:
        list of {
            'content':  str   — nội dung node/đoạn liên quan
            'score':    float — relevance score (0.0–1.0, ước tính từ rank)
            'metadata': dict  — {source, node_id, page_index, title}
            'source':   str   — 'pageindex'
        }
    """
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError(
            "Chưa set PAGEINDEX_API_KEY. Xem hướng dẫn trong upload_documents()."
        )

    from pageindex import PageIndexClient

    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_id_cache()

    if not cache:
        raise RuntimeError(
            "Chưa có document nào được upload. Hãy chạy upload_documents() trước."
        )

    doc_ids = list(cache.values())

    # --- Bước 1: Gọi PageIndex Chat API (agentic reasoning-based retrieval) ---
    try:
        response = pi.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
            enable_citations=True,
            temperature=0.0,
        )
        answer_text: str = response["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Lỗi khi gọi PageIndex Chat API: {e}") from e

    # --- Bước 2: Thu thập tree nodes từ tất cả documents để làm source ---
    all_nodes: list[dict] = []
    for md_filename, doc_id in cache.items():
        try:
            tree_result = pi.get_tree(doc_id, node_summary=False)
            if tree_result.get("status") != "completed":
                continue
            nodes = _flatten_tree(tree_result.get("result", []), source=md_filename)
            all_nodes.extend(nodes)
        except Exception:
            pass  # Bỏ qua nếu tree chưa sẵn sàng

    # --- Bước 3: Rank nodes theo relevance với query + answer ---
    #
    # PageIndex không trả về scored chunks như BM25 — thay vào đó ta dùng
    # simple keyword overlap để estimate relevance score cho từng tree node,
    # sau đó sort và lấy top_k.
    #
    # Đây là bước post-processing phía client; reasoning thực sự đã xảy ra
    # bên trong PageIndex khi gọi chat_completions() ở bước 1.

    query_tokens = set(query.lower().split())
    answer_tokens = set(answer_text.lower().split())
    combined_tokens = query_tokens | answer_tokens

    ranked = []
    for node in all_nodes:
        node_tokens = set(node["content"].lower().split())
        if not node_tokens:
            continue
        overlap = len(node_tokens & combined_tokens) / (len(node_tokens) ** 0.5 + 1e-9)
        ranked.append((overlap, node))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_nodes = ranked[:top_k]

    # Nếu không có node nào (tree chưa ready), trả về answer as single result
    if not top_nodes:
        return [
            {
                "content": answer_text,
                "score": 1.0,
                "metadata": {"source": "pageindex_chat", "node_id": None, "title": query},
                "source": "pageindex",
            }
        ]

    # Normalize scores về [0, 1]
    max_score = top_nodes[0][0] if top_nodes[0][0] > 0 else 1.0
    results = [
        {
            "content": node["content"],
            "score": round(score / max_score, 4),
            "metadata": node["metadata"],
            "source": "pageindex",
        }
        for score, node in top_nodes
    ]

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_tree(nodes: list[dict], source: str, depth: int = 0) -> list[dict]:
    """
    Flatten hierarchical PageIndex tree thành list phẳng của các node.

    Mỗi node trong tree có: title, node_id, page_index, text, nodes (children).
    """
    flat = []
    for node in nodes:
        content = node.get("text", "").strip()
        if content:
            flat.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source,
                        "node_id": node.get("node_id"),
                        "page_index": node.get("page_index"),
                        "title": node.get("title", ""),
                        "depth": depth,
                    },
                }
            )
        # Đệ quy vào children
        children = node.get("nodes", [])
        if children:
            flat.extend(_flatten_tree(children, source=source, depth=depth + 1))
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠  Hãy set PAGEINDEX_API_KEY trong file .env")
        print("   1. Đăng ký tại: https://pageindex.ai/")
        print("   2. Lấy API key tại: https://dash.pageindex.ai/api-keys")
        print("   3. Thêm vào .env: PAGEINDEX_API_KEY=pi-...")
        sys.exit(1)

    # --- Upload ---
    print("=" * 60)
    print("Bước 1: Upload documents lên PageIndex")
    print("=" * 60)
    doc_map = upload_documents()
    print(f"\n  Tổng số documents đã index: {len(doc_map)}")

    # --- Test query ---
    print("\n" + "=" * 60)
    print("Bước 2: Test vectorless search")
    print("=" * 60)

    test_queries = [
        "hình phạt sử dụng ma tuý",
        "tiền chất là gì theo Luật Phòng chống ma túy 2021",
        "các biện pháp cai nghiện bắt buộc",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        try:
            results = pageindex_search(q, top_k=3)
            for i, r in enumerate(results, 1):
                title = r["metadata"].get("title", "")
                source = r["metadata"].get("source", "")
                print(f"  [{i}] score={r['score']:.3f} | {source} | {title}")
                print(f"       {r['content'][:120]}...")
        except Exception as e:
            print(f"  ✗ Lỗi: {e}", file=sys.stderr)