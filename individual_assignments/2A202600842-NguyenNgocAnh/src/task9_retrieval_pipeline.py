"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

=============================================================================
KIẾN TRÚC PIPELINE

    Query
      │
      ├──→ Semantic Search (Task 5)  ─┐
      │     dense retrieval, top-20    │
      │                                ├──→ RRF Merge ──→ top-20 unique
      ├──→ Lexical Search (Task 6)   ─┘     (Task 7)
      │     BM25, top-20
      │
      ├──→ Cross-Encoder Rerank (Task 7) ──→ top-5 reranked
      │     chấm lại (query, doc) pair
      │
      └──→ Nếu best rerank_score < SCORE_THRESHOLD
              └──→ PageIndex Fallback (Task 8)
                    vectorless, reasoning-based retrieval

=============================================================================
LÝ DO LỰA CHỌN

Merge bằng RRF (Reciprocal Rank Fusion):
    - Gộp dense + sparse không cần normalize score (BM25 ≠ cosine scale)
    - Robust, đơn giản, k=60 là default từ paper Cormack 2009

Rerank bằng Cross-Encoder:
    - Chính xác hơn bi-encoder vì thấy cả cặp (query, doc)
    - Chỉ rerank top-20 → không quá chậm trên CPU

Fallback threshold = 0.3 (cross-encoder logit):
    - Score âm hoặc rất thấp → model cho rằng không liên quan
    - Khi đó PageIndex dùng LLM reasoning để tìm — phù hợp
      cho câu hỏi phức tạp mà BM25 + embedding đều bỏ sót
=============================================================================
"""

import sys
from pathlib import Path

# Support cả import relative (khi dùng như module) và absolute (khi chạy trực tiếp)
_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from task5_semantic_search      import semantic_search
from task6_lexical_search        import lexical_search
from task7_reranking             import rerank, rerank_rrf
from task8_pageindex_vectorless  import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Cross-encoder raw logit threshold:
# - Logit > 0  : model cho rằng doc relevant
# - Logit < 0  : không relevant
# - Threshold = -2 nghĩa là: dù kết quả có "hơi liên quan" cũng chấp nhận
#   trước khi fallback. Đặt thấp để PageIndex chỉ kích hoạt khi thực sự cần.
SCORE_THRESHOLD = -2.0
DEFAULT_TOP_K   = 5
RERANK_POOL     = 20    # số candidates đưa vào reranker


# =============================================================================
# PIPELINE
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query:           Câu truy vấn
        top_k:           Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng rerank_score tối thiểu.
                         Nếu best < threshold → fallback PageIndex.
        use_reranking:   Áp dụng cross-encoder reranking hay không.
        verbose:         In thông tin debug từng bước.

    Returns:
        List of {
            'content'         : str,
            'score'           : float,
            'metadata'        : dict,
            'retrieval_source': str   # 'hybrid' | 'pageindex'
        }
        Sorted by score descending.
    """

    # ── Step 1: Song song semantic + lexical ─────────────────────────────────
    pool = RERANK_POOL  # lấy nhiều hơn top_k để reranker có đủ candidates

    if verbose:
        print(f"  [Step 1] Semantic search top-{pool} ...")
    dense_results  = semantic_search(query, top_k=pool)

    if verbose:
        print(f"  [Step 1] Lexical search top-{pool} ...")
    sparse_results = lexical_search(query, top_k=pool)

    if verbose:
        print(f"  Dense: {len(dense_results)} | Sparse: {len(sparse_results)}")

    # ── Step 2: Merge bằng RRF ───────────────────────────────────────────────
    merged = rerank_rrf(
        ranked_lists=[dense_results, sparse_results],
        top_k=pool,
        k=60,
    )
    # Đánh dấu nguồn = hybrid
    for item in merged:
        item["retrieval_source"] = "hybrid"

    if verbose:
        print(f"  [Step 2] Merged (RRF): {len(merged)} unique candidates")

    if not merged:
        if verbose:
            print("  ⚠ Không có kết quả từ hybrid search → fallback PageIndex")
        return _pageindex_fallback(query, top_k, verbose)

    # ── Step 3: Rerank ───────────────────────────────────────────────────────
    if use_reranking:
        if verbose:
            print(f"  [Step 3] Cross-encoder reranking {len(merged)} → top-{top_k} ...")
        final_results = rerank(
            query      = query,
            candidates = merged,
            top_k      = top_k,
            method     = "cross_encoder",
        )
        # Giữ retrieval_source sau rerank
        for item in final_results:
            item.setdefault("retrieval_source", "hybrid")
    else:
        if verbose:
            print(f"  [Step 3] Reranking skipped")
        final_results = merged[:top_k]

    if verbose:
        best = final_results[0]["score"] if final_results else None
        print(f"  [Step 3] Best score: {best}")

    # ── Step 4: Kiểm tra threshold → fallback ────────────────────────────────
    best_score = final_results[0]["score"] if final_results else float("-inf")

    if best_score < score_threshold:
        if verbose:
            print(
                f"  ⚠ Best score ({best_score:.3f}) < threshold ({score_threshold}) "
                f"→ Fallback PageIndex"
            )
        return _pageindex_fallback(query, top_k, verbose)

    return final_results[:top_k]


def _pageindex_fallback(query: str, top_k: int, verbose: bool = False) -> list[dict]:
    """
    Fallback sang PageIndex vectorless RAG.
    Được gọi khi hybrid search không trả về kết quả đủ tốt.
    """
    if verbose:
        print(f"  [Fallback] Querying PageIndex ...")
    try:
        results = pageindex_search(query, top_k=top_k)
        for item in results:
            item["retrieval_source"] = "pageindex"
        if verbose:
            print(f"  [Fallback] PageIndex trả về {len(results)} kết quả")
        return results
    except Exception as e:
        if verbose:
            print(f"  [Fallback] PageIndex lỗi: {e}. Trả về empty.")
        return []


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 9 — Retrieval Pipeline")
    parser.add_argument("query",      nargs="?", default=None, help="Query để test")
    parser.add_argument("--no-rerank", action="store_true",   help="Tắt reranking")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--top-k",     type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý quy định về cai nghiện bắt buộc",
    ]

    queries = [args.query] if args.query else test_queries

    for q in queries:
        print(f"\n{'='*65}")
        print(f"Query: {q}")
        print('='*65)

        results = retrieve(
            query           = q,
            top_k           = args.top_k,
            score_threshold = args.threshold,
            use_reranking   = not args.no_rerank,
            verbose         = args.verbose,
        )

        if not results:
            print("  (Không có kết quả)")
            continue

        for i, r in enumerate(results, 1):
            src   = r.get("retrieval_source", "?")
            typ   = r.get("metadata", {}).get("type", "?")
            fname = r.get("metadata", {}).get("source", "?")
            score = r.get("score", 0)
            print(f"  [{i}] score={score:+.3f}  [{src}|{typ}]  {fname}")
            print(f"       {r['content'][:110].strip()}...")
