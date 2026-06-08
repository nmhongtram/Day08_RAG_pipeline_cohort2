"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Tương thích với Task 4: sentence-transformers/all-MiniLM-L6-v2 + ChromaDB
"""

from pathlib import Path
from functools import lru_cache

# Phải khớp với config trong Task 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH  = str(Path(__file__).parent.parent / "data" / "vectorstore")
COLLECTION_NAME = "drug_law_docs"


@lru_cache(maxsize=1)
def _get_model():
    """Load model một lần duy nhất, cache lại cho các lần gọi sau."""
    from sentence_transformers import SentenceTransformer
    print(f"[semantic_search] Loading model: {EMBEDDING_MODEL} ...")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    """Kết nối ChromaDB và trả về collection, cache lại."""
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (0–1, cao hơn = liên quan hơn)
            'metadata': dict     # source, type, chunk_index, ...
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model đã dùng ở Task 4
    model = _get_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,  # phải khớp với Task 4
    ).tolist()

    # Bước 2: Query ChromaDB — cosine similarity (đã cấu hình lúc tạo collection)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # tránh lỗi khi DB ít chunk hơn top_k
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Chuyển kết quả về format chuẩn, sorted descending by score
    # ChromaDB trả về distance (0 = giống nhau) → score = 1 - distance
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content":  doc,
            "score":    round(1 - dist, 4),
            "metadata": meta,
        })

    # ChromaDB đã sort theo distance tăng dần (= score giảm dần), nhưng sort lại cho chắc
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    test_queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ bị bắt vì sử dụng ma tuý",
        "cai nghiện bắt buộc",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = semantic_search(query, top_k=5)
        for i, r in enumerate(results, 1):
            src  = r["metadata"].get("source", "?")
            kind = r["metadata"].get("type", "?")
            print(f"  [{i}] score={r['score']:.4f}  [{kind}] {src}")
            print(f"       {r['content'][:120].strip()}...")