"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Tương thích với embedding model (all-MiniLM-L6-v2) và
      vector store (ChromaDB) đã dùng ở Task 4.

Cách hoạt động:
    1. Encode query bằng cùng model ở Task 4 (all-MiniLM-L6-v2, normalized)
    2. Gọi ChromaDB query với cosine distance
    3. Chuyển distance → similarity: score = 1 - distance
    4. Trả về top_k kết quả sorted descending theo score
"""

from pathlib import Path

# Dùng lại constants từ task4 để đảm bảo nhất quán
_BASE_DIR       = Path(__file__).parent.parent
VECTORDB_DIR    = _BASE_DIR / "data" / "vectordb"
COLLECTION_NAME = "DrugLawDocs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache model/client để tránh reload mỗi lần gọi
_model      = None
_collection = None


def _get_model():
    """Load và cache SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Kết nối và cache ChromaDB collection."""
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(VECTORDB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (ChromaDB cosine).

    Args:
        query:  Câu truy vấn (tiếng Việt hoặc tiếng Anh)
        top_k:  Số lượng kết quả tối đa trả về

    Returns:
        List of {
            'content' : str,    # Nội dung chunk
            'score'   : float,  # Cosine similarity  ∈ [0, 1], cao hơn = tốt hơn
            'metadata': dict    # source, type, filepath, chunk_index, chunk_total
        }
        Sorted by score descending.
    """
    # --- Bước 1: Embed query (normalize để cosine = dot product) ---
    model = _get_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # --- Bước 2: Query ChromaDB ---
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # tránh lỗi nếu DB nhỏ
        include=["documents", "metadatas", "distances"],
    )

    # --- Bước 3: Chuyển distance → similarity và build output ---
    # ChromaDB cosine: distance = 1 - cosine_similarity  →  similarity = 1 - distance
    docs       = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        output.append({
            "content" : doc,
            "score"   : round(1.0 - dist, 6),   # cosine similarity
            "metadata": meta,
        })

    # Đảm bảo sorted descending (ChromaDB đã sort nhưng chắc chắn hơn)
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    import sys

    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ Việt Nam liên quan ma tuý",
        "cai nghiện bắt buộc",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        results = semantic_search(q, top_k=5)
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            typ = r["metadata"].get("type", "?")
            print(f"  [{i}] score={r['score']:.4f}  [{typ}] {src}")
            print(f"       {r['content'][:120].strip()}...")
