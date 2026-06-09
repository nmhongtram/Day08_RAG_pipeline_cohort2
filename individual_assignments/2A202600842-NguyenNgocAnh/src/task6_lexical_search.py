"""
Task 6 — Lexical Search Module (BM25).

Sử dụng BM25Okapi từ thư viện rank-bm25.

=============================================================================
BM25 hoạt động thế nào:
    BM25 (Best Match 25) là thuật toán ranking dựa trên bag-of-words:

    score(q, d) = Σ_i  IDF(qi) * tf(qi,d) * (k1 + 1)
                        ─────────────────────────────────
                        tf(qi,d) + k1 * (1 - b + b * |d|/avgdl)

    Trong đó:
    - tf(qi, d)  : số lần từ qi xuất hiện trong document d
    - IDF(qi)    : log((N - df + 0.5) / (df + 0.5) + 1)
                   N = tổng docs, df = số docs chứa qi
                   → từ hiếm có IDF cao → đóng góp điểm nhiều hơn
    - |d|        : độ dài document (số token)
    - avgdl      : độ dài trung bình của corpus
    - k1 = 1.5   : term saturation — giới hạn ảnh hưởng của TF
                   (tf lớn vẫn không tăng điểm vô hạn)
    - b  = 0.75  : length normalization — document dài không được ưu tiên quá mức

    So sánh với TF-IDF:
    TF-IDF đơn giản là TF × IDF, không có saturation và length norm.
    BM25 tốt hơn nhờ: (1) term saturation tránh document nhồi từ khóa,
    (2) length normalization cân bằng giữa doc ngắn và dài.

    Tokenization (tiếng Việt):
    Dùng simple whitespace split + lowercase, có thể cải thiện bằng
    underthesea/pyvi để word-segment chính xác hơn.
=============================================================================

Cài đặt:
    pip install rank-bm25   (đã có từ crawl4ai deps)
"""

from pathlib import Path

_BASE_DIR       = Path(__file__).parent.parent
VECTORDB_DIR    = _BASE_DIR / "data" / "vectordb"
COLLECTION_NAME = "DrugLawDocs"

# Cache BM25 index và corpus sau lần build đầu tiên
_bm25   = None
_corpus: list[dict] = []


# =============================================================================
# Tokenizer
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Tokenize text cho BM25.

    Chiến lược đơn giản: lowercase + split by whitespace.
    Đủ tốt cho văn bản pháp luật và báo tiếng Việt ở mức demo.
    Cải thiện: dùng underthesea.word_tokenize() cho word-segmentation
    chính xác hơn nhưng cần cài thêm ~500MB model.
    """
    import re
    # Lowercase + giữ lại chữ, số, khoảng trắng (bỏ ký tự đặc biệt)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


# =============================================================================
# Load corpus từ ChromaDB (dùng lại data đã index ở Task 4)
# =============================================================================

def _load_corpus_from_chromadb() -> list[dict]:
    """
    Lấy toàn bộ chunks từ ChromaDB collection.
    Dùng lại data đã index ở Task 4 — không đọc lại file markdown.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()

    # ChromaDB get() có giới hạn 1000 mặc định, dùng offset để lấy hết
    corpus = []
    BATCH = 500
    for offset in range(0, total, BATCH):
        res = collection.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(res["documents"], res["metadatas"]):
            corpus.append({"content": doc, "metadata": meta})

    return corpus


# =============================================================================
# Build BM25 index
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25Okapi index từ corpus.

    BM25Okapi dùng k1=1.5, b=0.75 mặc định — phù hợp cho văn bản chung.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance
    """
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25


def _ensure_index():
    """Đảm bảo BM25 index và corpus đã được khởi tạo (lazy init)."""
    global _bm25, _corpus
    if _bm25 is None:
        _corpus = _load_corpus_from_chromadb()
        _bm25   = build_bm25_index(_corpus)


# =============================================================================
# Lexical Search
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25Okapi.

    Args:
        query:  Câu truy vấn (tiếng Việt hoặc tiếng Anh)
        top_k:  Số lượng kết quả tối đa

    Returns:
        List of {
            'content' : str,    # Nội dung chunk
            'score'   : float,  # BM25 score (không normalize, cao hơn = tốt hơn)
            'metadata': dict    # source, type, filepath, chunk_index
        }
        Sorted by score descending. Chỉ trả về chunks có score > 0.
    """
    import numpy as np

    _ensure_index()

    # Tokenize query cùng chiến lược với corpus
    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)  # numpy array, len = len(_corpus)

    # Lấy top_k indices sorted descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        s = float(scores[idx])
        if s <= 0:
            continue  # bỏ qua chunk không liên quan
        results.append({
            "content" : _corpus[idx]["content"],
            "score"   : round(s, 6),
            "metadata": _corpus[idx]["metadata"],
        })

    # Đã sorted descending nhờ argsort nhưng đảm bảo chắc chắn
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import sys

    queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "nghệ sĩ bị khởi tố ma tuý",
        "cai nghiện bắt buộc hồ sơ",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    print("Đang khởi tạo BM25 index ...")
    _ensure_index()
    print(f"✓ Corpus: {len(_corpus)} chunks\n")

    for q in queries:
        print(f"{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        results = lexical_search(q, top_k=5)
        if not results:
            print("  (Không có kết quả)")
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            typ = r["metadata"].get("type", "?")
            print(f"  [{i}] score={r['score']:.4f}  [{typ}] {src}")
            print(f"       {r['content'][:120].strip()}...")
        print()
