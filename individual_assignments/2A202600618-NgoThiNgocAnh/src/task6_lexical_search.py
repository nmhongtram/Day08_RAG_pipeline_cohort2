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

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# TOKENIZER
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize tiếng Việt đơn giản:
      - Lowercase
      - Tách theo khoảng trắng và dấu câu
      - Lọc bỏ token rỗng / chỉ chứa số đơn lẻ

    Nếu muốn độ chính xác cao hơn, thay bằng:
        from underthesea import word_tokenize
        return word_tokenize(text.lower(), format="text").split()
    """
    text = text.lower()
    # Giữ lại chữ cái, số, khoảng trắng; bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


# =============================================================================
# LOAD CORPUS
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Đọc toàn bộ .md files từ data/standardized/ làm corpus cho BM25.
    Mỗi file = 1 document (không chunk lại — BM25 hoạt động tốt trên đoạn dài).
    """
    corpus = []
    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Không tìm thấy: {STANDARDIZED_DIR}. Hãy chạy Task 3 trước.")
        return corpus

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file),
            },
        })
    return corpus


# =============================================================================
# BM25 INDEX (lazy, cached)
# =============================================================================

_bm25_index: BM25Okapi | None = None
_corpus: list[dict] = []


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance đã được index
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    # BM25Okapi: k1=1.5 (term saturation), b=0.75 (length normalization)
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    return bm25


def _ensure_index():
    """Load corpus và build index nếu chưa có (lazy init)."""
    global _bm25_index, _corpus
    if _bm25_index is None:
        _corpus = load_corpus()
        if not _corpus:
            raise RuntimeError("Corpus rỗng — hãy chạy Task 3 trước để tạo data/standardized/")
        print(f"[bm25] Building index trên {len(_corpus)} documents ...")
        _bm25_index = build_bm25_index(_corpus)
        print("[bm25] Index ready.")


# =============================================================================
# LEXICAL SEARCH
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,   # BM25 score (không có upper bound — giá trị tương đối)
            'metadata': dict  # source, type, path
        }
        Sorted by score descending. Chỉ trả về kết quả có score > 0.
    """
    _ensure_index()

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = _bm25_index.get_scores(tokenized_query)

    # Lấy top_k index, sort descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # bỏ document không khớp từ nào
        results.append({
            "content":  _corpus[idx]["content"],
            "score":    float(round(scores[idx], 4)),
            "metadata": _corpus[idx]["metadata"],
        })

    return results  # đã sorted descending by score


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "nghệ sĩ ca sĩ bị bắt sử dụng ma tuý",
        "hình phạt tù cai nghiện bắt buộc",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query : {query}")
        print("=" * 60)
        results = lexical_search(query, top_k=5)
        if not results:
            print("  (không có kết quả)")
        for i, r in enumerate(results, 1):
            src  = r["metadata"].get("source", "?")
            kind = r["metadata"].get("type", "?")
            print(f"  [{i}] score={r['score']:.4f}  [{kind}] {src}")
            print(f"       {r['content'][:120].strip()} ...")