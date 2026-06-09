"""
Task 7 — Reranking Module.

Phương pháp chính: Cross-encoder reranker (local model).

=============================================================================
TẠI SAO CHỌN CROSS-ENCODER?

Bi-encoder (Task 5):
    query → encoder → vector q
    doc   → encoder → vector d
    score = cosine(q, d)        ← encode độc lập, nhanh nhưng kém chính xác

Cross-encoder:
    [query, doc] → encoder → relevance score
    Model thấy TOÀN BỘ cặp (query, doc) cùng lúc, dùng attention để hiểu
    mối quan hệ giữa từng token của query với từng token của document.
    → Chính xác hơn nhiều, nhưng chậm hơn (không cache được embedding).
    → Chỉ dùng để rerank top-N candidates, không scan toàn bộ corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Nhẹ (~85MB), chạy tốt trên CPU
    - Được fine-tune trên MS MARCO passage ranking
    - Hỗ trợ tiếng Anh tốt, tiếng Việt ở mức chấp nhận được cho demo
    - Nếu cần multilingual chính xác hơn:
      * jinaai/jina-reranker-v2-base-multilingual (API hoặc local)
      * Qwen/Qwen3-Reranker-0.6B (local, nhẹ)
      * BAAI/bge-reranker-v2-m3 (local, multilingual tốt)

Pipeline rerank:
    candidates (top-20 từ retrieval)
         ↓  cross-encoder score mỗi cặp (query, doc)
    reranked (top-5 sorted by cross-encoder score)
=============================================================================
"""

from typing import Optional
import numpy as np

_BASE_DIR       = __import__("pathlib").Path(__file__).parent.parent

# Model cross-encoder — dùng MiniLM nhẹ, chạy được trên CPU
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache cross-encoder model
_cross_encoder = None


# =============================================================================
# Helper: load cross-encoder (lazy)
# =============================================================================

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL} ...")
        # max_length=512 — đủ cho một chunk văn bản pháp luật
        _cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL,
            max_length=512,
            device="cpu",
        )
    return _cross_encoder


# =============================================================================
# Helper: cosine similarity
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder local model.

    Model đánh giá mức độ liên quan của cặp (query, document) bằng cách
    cho attention cross giữa các token của query và document — chính xác
    hơn bi-encoder vì thấy toàn bộ context của cả hai.

    Args:
        query:      Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k:      Số lượng kết quả sau rerank

    Returns:
        top_k candidates sorted by rerank_score descending.
        Mỗi item có thêm key 'rerank_score' (raw logit) và
        'original_score' (score gốc từ retrieval).
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Tạo pairs (query, document_content) cho cross-encoder
    pairs = [(query, c["content"]) for c in candidates]

    # Predict scores — trả về raw logit (không normalize)
    # Dùng activation_fct=None để giữ raw logit, dễ so sánh tương đối
    scores = model.predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Build output — giữ lại score gốc, thêm rerank_score
    reranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["original_score"] = candidate.get("score", 0.0)
        item["rerank_score"]   = float(score)
        item["score"]          = float(score)   # override score = rerank score
        reranked.append(item)

    # Sort by rerank_score descending, lấy top_k
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn vừa relevant vừa diverse.

    Công thức:
        MMR(d) = λ * sim(query, d) - (1 - λ) * max_{s ∈ S} sim(d, s)

    Giải thích:
        - sim(query, d): độ liên quan của d với query (relevance)
        - max sim(d, s): độ tương đồng với các doc đã chọn (redundancy)
        - λ = 0.7: ưu tiên relevance hơn diversity (0=diversity only, 1=relevance only)
        - Greedy: mỗi bước chọn doc có MMR score cao nhất chưa được chọn

    Args:
        query_embedding: Vector embedding của query (normalized)
        candidates:      List có 'embedding' key
        top_k:           Số kết quả cần chọn
        lambda_param:    Trade-off relevance vs diversity ∈ [0, 1]

    Returns:
        top_k candidates theo MMR, mỗi item có thêm 'mmr_score'.
    """
    if not candidates:
        return []

    n = min(top_k, len(candidates))
    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(n):
        best_idx   = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            if not emb:
                # Nếu không có embedding, dùng score gốc làm relevance
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = _cosine_sim(query_embedding, emb)

            # Max similarity với các doc đã chọn
            redundancy = 0.0
            for sel_idx in selected_indices:
                sel_emb = candidates[sel_idx].get("embedding", [])
                if emb and sel_emb:
                    sim = _cosine_sim(emb, sel_emb)
                    redundancy = max(redundancy, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr > best_score:
                best_score = mmr
                best_idx   = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for rank, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        item["mmr_score"] = rank  # thứ tự MMR selection
        results.append(item)

    return results


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Công thức (Cormack et al. 2009):
        RRF(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))

    Giải thích:
        - Mỗi ranker cho doc d một rank (1-indexed)
        - 1/(k+rank) giảm nhanh với rank thấp, k=60 làm "mượt" sự chênh lệch
        - Doc xuất hiện nhiều ranker VÀ rank cao → RRF score cao
        - Không cần normalize score giữa các ranker (BM25 ≠ cosine scale)
        - k=60: từ paper gốc, thực nghiệm tốt cho nhiều bài toán

    Ưu điểm: Đơn giản, không cần thêm model, robust với outlier score.
    Nhược: Không dùng được magnitude của score gốc.

    Args:
        ranked_lists: List của các ranked list (mỗi list từ 1 retrieval method)
        top_k:        Số kết quả cuối
        k:            Smoothing constant (default=60)

    Returns:
        top_k items sorted by RRF score descending, có thêm key 'rrf_score'.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"][:200]  # dùng 200 chars đầu làm key
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in sorted_items[:top_k]:
        item = content_map[key].copy()
        item["rrf_score"] = round(score, 6)
        item["score"]     = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
    query_embedding: Optional[list[float]] = None,
    ranked_lists: Optional[list[list[dict]]] = None,
    lambda_param: float = 0.7,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query:           Câu truy vấn
        candidates:      Candidates từ retrieval (dùng cho cross_encoder, mmr)
        top_k:           Số kết quả sau rerank
        method:          'cross_encoder' | 'mmr' | 'rrf'
        query_embedding: Cần cho method='mmr'
        ranked_lists:    Cần cho method='rrf' (thay thế candidates)
        lambda_param:    MMR trade-off parameter
        rrf_k:           RRF smoothing constant

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("method='mmr' cần query_embedding")
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        lists = ranked_lists if ranked_lists is not None else [candidates]
        return rerank_rrf(lists, top_k, rrf_k)

    else:
        raise ValueError(f"Unknown method: {method}. Chọn: cross_encoder | mmr | rrf")


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import sys
    from task5_semantic_search import semantic_search
    from task6_lexical_search  import lexical_search

    query = "hình phạt tội tàng trữ ma tuý"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print(f"Query: {query}\n")

    # Lấy candidates từ semantic + lexical (top 10 mỗi loại → dedupe → 15 unique)
    sem_results = semantic_search(query, top_k=10)
    lex_results = lexical_search(query, top_k=10)

    # Deduplicate theo content
    seen = set()
    candidates = []
    for r in sem_results + lex_results:
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            candidates.append(r)
    candidates = candidates[:15]

    print(f"Candidates từ retrieval: {len(candidates)}")

    # ── Cross-encoder rerank ──
    print(f"\n{'='*60}")
    print("Cross-Encoder Rerank (top 5)")
    print('='*60)
    reranked = rerank(query, candidates, top_k=5, method="cross_encoder")
    for i, r in enumerate(reranked, 1):
        src = r["metadata"].get("source", "?")
        typ = r["metadata"].get("type", "?")
        print(f"  [{i}] rerank={r['rerank_score']:+.3f}  orig={r['original_score']:.3f}"
              f"  [{typ}] {src}")
        print(f"       {r['content'][:110].strip()}...")

    # ── RRF (bonus: so sánh) ──
    print(f"\n{'='*60}")
    print("RRF Rerank (top 5) — semantic + lexical fusion")
    print('='*60)
    rrf_results = rerank(
        query, candidates, top_k=5,
        method="rrf",
        ranked_lists=[sem_results, lex_results],
    )
    for i, r in enumerate(rrf_results, 1):
        src = r["metadata"].get("source", "?")
        typ = r["metadata"].get("type", "?")
        print(f"  [{i}] rrf={r['rrf_score']:.5f}  [{typ}] {src}")
        print(f"       {r['content'][:110].strip()}...")
