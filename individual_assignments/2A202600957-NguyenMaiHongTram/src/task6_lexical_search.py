"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
from pathlib import Path

from .task4_chunking_indexing import chunk_documents, load_documents

# Corpus = tất cả chunks tạo ra ở Task 4 (cùng nguồn dữ liệu, cùng ranh giới
# chunk với semantic search, để hybrid search ở Task 9 so sánh "táo với táo").
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_bm25_index = None

# Re-chunk toàn bộ corpus bằng SemanticChunker khá tốn thời gian (phải embed
# từng câu). Cache kết quả ra đĩa để các lần chạy sau (và các test khác trong
# cùng phiên pytest) tái sử dụng thay vì build lại từ đầu.
_CORPUS_CACHE_PATH = Path(__file__).parent.parent / "data" / "bm25_corpus_cache.json"


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase + split theo khoảng trắng."""
    return text.lower().split()


def _ensure_corpus() -> list[dict]:
    """
    Lazy-load CORPUS từ data/standardized/ (qua cùng pipeline chunk ở Task 4),
    có cache ra đĩa (data/bm25_corpus_cache.json) vì semantic chunking toàn bộ
    corpus khá chậm (phải embed từng câu để dò ranh giới ngữ nghĩa).
    """
    global CORPUS
    if CORPUS:
        return CORPUS

    if _CORPUS_CACHE_PATH.exists():
        CORPUS = json.loads(_CORPUS_CACHE_PATH.read_text(encoding="utf-8"))
        return CORPUS

    CORPUS = chunk_documents(load_documents())
    try:
        _CORPUS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CORPUS_CACHE_PATH.write_text(
            json.dumps(CORPUS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index đã được build trên corpus đã tokenize.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index

    corpus = _ensure_corpus()
    if not corpus:
        return []

    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)

    import numpy as np

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
