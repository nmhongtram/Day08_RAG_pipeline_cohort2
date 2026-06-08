"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .embeddings import embed_query
from .task4_chunking_indexing import WEAVIATE_COLLECTION


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (Weaviate `near_vector`).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import weaviate
    from weaviate.classes.query import MetadataQuery

    # Bước 1: Embed query bằng cùng model BAAI/bge-m3 dùng ở Task 4
    query_embedding = embed_query(query)

    # Bước 2 & 3: Query vector store (cosine similarity) và lấy top_k.
    # Weaviate có thể không sẵn sàng (không chạy local/Docker) ở môi trường
    # dev — graceful degradation: trả về [] kèm cảnh báo thay vì crash, để
    # pipeline (Task 9) vẫn hoạt động được qua nhánh lexical/PageIndex.
    try:
        client = weaviate.connect_to_local()
    except Exception as e:
        print(f"  ⚠ semantic_search: không kết nối được Weaviate ({e}) — trả về []")
        return []

    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            results.append({
                "content": props.get("content", ""),
                "score": 1.0 - distance,  # cosine distance -> similarity
                "metadata": {
                    "source": props.get("source", ""),
                    "type": props.get("doc_type", ""),
                    "chunk_index": props.get("chunk_index", 0),
                },
            })
    except Exception as e:
        print(f"  ⚠ semantic_search: lỗi truy vấn Weaviate ({e}) — trả về []")
        return []
    finally:
        client.close()

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
