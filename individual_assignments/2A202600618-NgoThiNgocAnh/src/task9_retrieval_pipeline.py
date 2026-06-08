"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""


from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # ===== Step 1: Retrieve candidates =====
    try:
        dense_hits = semantic_search(query, top_k=top_k * 2)
    except Exception as err:
        print(f"  ⚠ Dense search failed: {err}")
        dense_hits = []

    try:
        sparse_hits = lexical_search(query, top_k=top_k * 2)
    except Exception as err:
        print(f"  ⚠ Sparse search failed: {err}")
        sparse_hits = []

    # ===== Step 2: Fusion (RRF) =====
    combined = rerank_rrf([dense_hits, sparse_hits], top_k=top_k * 2)

    for doc in combined:
        doc["source"] = "hybrid"

    # ===== Step 3: Reranking =====
    if use_reranking and combined:
        try:
            ranked = rerank(query, combined, top_k=top_k, method=RERANK_TYPE)
        except Exception as err:
            print(f"  ⚠ Rerank failed: {err}")
            ranked = combined[:top_k]
    else:
        ranked = combined[:top_k]

    for doc in ranked:
        doc.setdefault("source", "hybrid")

    # ===== Step 4: Fallback nếu score thấp =====
    if not ranked or ranked[0]["score"] < score_threshold:
        current_score = ranked[0]["score"] if ranked else 0.0
        print(f"  ⚠ Low score ({current_score:.3f}) → switching to PageIndex")

        try:
            alt_results = pageindex_search(query, top_k=top_k)
            if alt_results:
                return alt_results[:top_k]
        except Exception as err:
            print(f"  ⚠ PageIndex error: {err}")

    # ===== Step 5: Return final =====
    return ranked[:top_k]



if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
