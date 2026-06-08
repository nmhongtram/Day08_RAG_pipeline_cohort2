"""
Task 7 — Reranking Module.

Implement cả 3 phương pháp:
    - RRF  (Reciprocal Rank Fusion)  ← default, không cần thêm thư viện
    - MMR  (Maximal Marginal Relevance)
    - Cross-encoder (Jina Reranker API)

Cài đặt:
    pip install numpy requests          # RRF + MMR + Jina API
    pip install sentence-transformers   # đã có từ Task 4/5 (dùng để lấy query embedding cho MMR)
"""

import os
from typing import Optional

import numpy as np


# =============================================================================
# HELPER
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# =============================================================================
# METHOD 1: RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Cơ chế:
        RRF(d) = Σ 1 / (k + rank_r(d))
        - Mỗi document nhận điểm từ vị trí của nó trong từng ranked list.
        - Document xuất hiện ở rank cao trong nhiều list → tổng điểm cao.
        - k=60 (từ paper Cormack et al. 2009) làm giảm ảnh hưởng của rank 1
          so với rank 2, tránh 1 ranker chi phối quá mức.
        - Không cần normalize score giữa các ranker → robust khi kết hợp
          BM25 (score tuyệt đối) với cosine similarity (score 0-1).

    Args:
        ranked_lists: List của các ranked result lists (mỗi list từ 1 ranker).
                      Mỗi item cần có key 'content'.
        top_k: Số lượng kết quả cuối cùng.
        k: Smoothing constant (default=60).

    Returns:
        List of top_k candidates sorted by RRF score descending,
        mỗi item có thêm key 'rrf_score'.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict]  = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, rrf_score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(rrf_score, 6)
        item["rrf_score"] = round(rrf_score, 6)
        results.append(item)

    return results


# =============================================================================
# METHOD 2: MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Cơ chế:
        MMR(d) = λ * sim(query, d) - (1-λ) * max_sim(d, selected)
        - Vòng lặp greedy: mỗi bước chọn document tối đa hoá MMR score.
        - λ=1.0 → thuần relevance (giống top-k thường).
        - λ=0.0 → thuần diversity (tránh trùng lặp tối đa).
        - λ=0.7 → cân bằng: ưu tiên relevance nhưng vẫn đa dạng.
        - Hữu ích khi corpus có nhiều chunk gần giống nhau (ví dụ cùng điều luật).

    Args:
        query_embedding: Vector embedding của query (cùng model Task 4/5).
        candidates: List of {'content', 'score', 'embedding', 'metadata'}.
                    Cần có key 'embedding' — nếu không có, dùng rerank_rrf.
        top_k: Số lượng kết quả.
        lambda_param: Trade-off relevance (1.0) vs diversity (0.0).

    Returns:
        List of top_k candidates selected by MMR, có thêm key 'mmr_score'.
    """
    if not candidates:
        return []

    # Kiểm tra embedding tồn tại
    if "embedding" not in candidates[0]:
        raise ValueError(
            "Candidates thiếu key 'embedding'. "
            "Dùng rerank_rrf nếu không có embedding, "
            "hoặc thêm embedding cho candidates trước khi gọi MMR."
        )

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            # Relevance: cosine sim giữa query và candidate
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Redundancy: cosine sim lớn nhất với các doc đã chọn
            if selected_indices:
                max_sim = max(
                    _cosine_sim(candidates[idx]["embedding"], candidates[s]["embedding"])
                    for s in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    results = []
    for i, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        # Tính lại mmr_score để lưu vào output
        relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])
        item["mmr_score"] = round(relevance, 4)   # dùng relevance làm score hiển thị
        item["score"] = item["mmr_score"]
        results.append(item)

    return results


# =============================================================================
# METHOD 3: Cross-encoder — Jina Reranker API
# =============================================================================

JINA_API_KEY = os.getenv("JINA_API_KEY", "")  # export JINA_API_KEY=jina_...


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker v2 (multilingual, hỗ trợ tiếng Việt).

    Cơ chế:
        Cross-encoder nhận cặp (query, document) và tính relevance score trực tiếp
        thay vì so sánh 2 vector riêng lẻ như bi-encoder.
        → Chính xác hơn bi-encoder nhưng chậm hơn (O(n) inference calls).
        → Thường dùng ở bước rerank sau khi đã lọc còn ~20-50 candidates.

    Đăng ký API key miễn phí tại: https://jina.ai (1M tokens/tháng free)
    Set biến môi trường: export JINA_API_KEY=jina_...

    Args:
        query: Câu truy vấn.
        candidates: List of {'content', 'score', 'metadata'}.
        top_k: Số lượng kết quả sau rerank.

    Returns:
        List of top_k candidates sorted by relevance_score descending.
    """
    if not JINA_API_KEY:
        raise EnvironmentError(
            "Thiếu JINA_API_KEY. "
            "Đăng ký miễn phí tại https://jina.ai rồi:\n"
            "  Windows: set JINA_API_KEY=jina_...\n"
            "  Linux/Mac: export JINA_API_KEY=jina_..."
        )

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]
    return [
        {
            **candidates[r["index"]],
            "score": round(r["relevance_score"], 4),
            "rerank_score": round(r["relevance_score"], 4),
        }
        for r in reranked
    ]


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",          # "rrf" | "mmr" | "cross_encoder"
    semantic_results: Optional[list[dict]] = None,   # dùng khi method="rrf"
    lexical_results: Optional[list[dict]] = None,    # dùng khi method="rrf"
    query_embedding: Optional[list[float]] = None,   # dùng khi method="mmr"
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface.

    Cách dùng phổ biến nhất (RRF kết hợp semantic + lexical):
        results = rerank(
            query="hình phạt tàng trữ ma tuý",
            candidates=[],          # không dùng khi method="rrf"
            method="rrf",
            semantic_results=semantic_search(query, top_k=20),
            lexical_results=lexical_search(query, top_k=20),
            top_k=5,
        )
    """
    if method == "rrf":
        lists = []
        if semantic_results:
            lists.append(semantic_results)
        if lexical_results:
            lists.append(lexical_results)
        if candidates and not lists:
            lists.append(candidates)
        if not lists:
            raise ValueError("RRF cần ít nhất 1 ranked list (semantic_results hoặc lexical_results).")
        return rerank_rrf(lists, top_k=top_k)

    elif method == "mmr":
        if query_embedding is None:
            # Tự embed query nếu không truyền vào
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-m3")
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(query_embedding, candidates, top_k=top_k, lambda_param=lambda_param)

    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}. Chọn: 'rrf' | 'mmr' | 'cross_encoder'")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # ── Test RRF (không cần thư viện thêm) ──────────────────────────────────
    print("=" * 60)
    print("TEST 1: RRF — kết hợp semantic + lexical results")
    print("=" * 60)

    # Giả lập output từ Task 5 (semantic) và Task 6 (lexical)
    semantic_results = [
        {"content": "Điều 249: Tội sản xuất trái phép chất ma tuý, phạt tù 2-7 năm", "score": 0.91, "metadata": {"source": "luat.md", "type": "legal"}},
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý, phạt tù 1-5 năm", "score": 0.87, "metadata": {"source": "luat.md", "type": "legal"}},
        {"content": "Ca sĩ X bị bắt tại nhà riêng với tang vật ma tuý", "score": 0.72, "metadata": {"source": "article_01.md", "type": "news"}},
    ]
    lexical_results = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý, phạt tù 1-5 năm", "score": 8.5, "metadata": {"source": "luat.md", "type": "legal"}},
        {"content": "Hình phạt bổ sung: phạt tiền 5-500 triệu đồng cho tội tàng trữ", "score": 7.2, "metadata": {"source": "luat.md", "type": "legal"}},
        {"content": "Ca sĩ X bị bắt tại nhà riêng với tang vật ma tuý", "score": 5.1, "metadata": {"source": "article_01.md", "type": "news"}},
    ]

    rrf_results = rerank(
        query="hình phạt tàng trữ ma tuý",
        candidates=[],
        method="rrf",
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        top_k=3,
    )
    for i, r in enumerate(rrf_results, 1):
        print(f"  [{i}] rrf_score={r['score']:.6f} | {r['content'][:80]}")

    # ── Test MMR ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: MMR — diversity reranking (dùng dummy embeddings)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    base = rng.random(8).tolist()

    mmr_candidates = [
        {"content": "Điều 248: tàng trữ ma tuý phạt tù 1-5 năm",        "score": 0.87, "metadata": {}, "embedding": (np.array(base) + rng.random(8) * 0.1).tolist()},
        {"content": "Điều 248 khoản 2: tàng trữ số lượng lớn phạt 5-10 năm", "score": 0.85, "metadata": {}, "embedding": (np.array(base) + rng.random(8) * 0.1).tolist()},
        {"content": "Ca sĩ Y sử dụng ma tuý bị khởi tố",                "score": 0.75, "metadata": {}, "embedding": rng.random(8).tolist()},
        {"content": "Hình phạt bổ sung: phạt tiền 5-500 triệu",          "score": 0.70, "metadata": {}, "embedding": rng.random(8).tolist()},
    ]
    query_emb = rng.random(8).tolist()

    mmr_results = rerank(
        query="hình phạt tàng trữ ma tuý",
        candidates=mmr_candidates,
        method="mmr",
        query_embedding=query_emb,
        lambda_param=0.7,
        top_k=3,
    )
    for i, r in enumerate(mmr_results, 1):
        print(f"  [{i}] mmr_score={r['score']:.4f} | {r['content'][:80]}")

    print("\n✓ RRF và MMR hoạt động bình thường.")
    print("  Cross-encoder: set JINA_API_KEY để test (đăng ký miễn phí tại jina.ai)")