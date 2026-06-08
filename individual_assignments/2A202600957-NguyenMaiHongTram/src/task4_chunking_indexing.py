"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

from .embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    FastBreakpointEmbeddings,
    embed_texts,
)

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

WEAVIATE_COLLECTION = "DrugLawDocs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: SemanticChunker
#   Văn bản pháp luật có các điều/khoản dài, ranh giới ngữ nghĩa không trùng
#   với ranh giới ký tự cố định → tách theo ngữ nghĩa (similarity giữa các
#   câu liền kề) giữ nguyên vẹn ý nghĩa của từng điều khoản/đoạn tin tức,
#   tốt hơn cắt cứng theo số ký tự.
# CHUNK_SIZE / CHUNK_OVERLAP: dùng làm ngưỡng "chốt chặn" — nếu 1 đoạn ngữ
#   nghĩa quá dài (vượt CHUNK_SIZE), ta cắt tiếp bằng RecursiveCharacterText
#   Splitter với overlap=CHUNK_OVERLAP để không vượt giới hạn kích thước
#   (đảm bảo chunk không quá dài cho embedding model / context window) trong
#   khi vẫn giữ được câu liền mạch nhờ overlap.
CHUNK_SIZE = 500        # Đủ nhỏ để mỗi chunk tập trung 1 ý, vừa context LLM
CHUNK_OVERLAP = 50      # ~10% CHUNK_SIZE — giữ liên kết ngữ cảnh giữa 2 chunk liền kề
CHUNKING_METHOD = "semantic"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: BAAI/bge-m3 — multilingual, 1024-dim, tốt cho tiếng Việt
# (xem giải thích chi tiết trong src/embeddings.py)
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME  # "BAAI/bge-m3"
EMBEDDING_DIM = EMBEDDING_DIM           # 1024

# Vector store: Weaviate — hỗ trợ hybrid search (dense + BM25) built-in,
# vector thuần (Configure.Vectorizer.none()) vì ta tự embed bằng BGE-M3
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def _split_oversized(text: str) -> list[str]:
    """Cắt tiếp 1 đoạn văn bản quá dài thành các phần <= CHUNK_SIZE."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _chunk_semantic(content: str) -> list[str]:
    """
    Tách văn bản theo ranh giới ngữ nghĩa bằng SemanticChunker (BGE-M3),
    sau đó "chốt chặn" kích thước: đoạn nào > CHUNK_SIZE sẽ được cắt tiếp
    bằng RecursiveCharacterTextSplitter (giữ overlap để không mất ngữ cảnh).
    """
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        FastBreakpointEmbeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    semantic_splits = splitter.split_text(content)

    final_splits = []
    for split in semantic_splits:
        if len(split) <= CHUNK_SIZE:
            final_splits.append(split)
        else:
            final_splits.extend(_split_oversized(split))
    return final_splits


def _chunk_recursive(content: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(content)


def _chunk_markdown_header(content: str) -> list[str]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(content)

    final_splits = []
    for doc in docs:
        text = doc.page_content
        if len(text) <= CHUNK_SIZE:
            final_splits.append(text)
        else:
            final_splits.extend(_split_oversized(text))
    return final_splits


_CHUNKERS = {
    "semantic": _chunk_semantic,
    "recursive": _chunk_recursive,
    "markdown_header": _chunk_markdown_header,
}


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (CHUNKING_METHOD).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunk_fn = _CHUNKERS[CHUNKING_METHOD]

    chunks = []
    for doc in documents:
        splits = chunk_fn(doc["content"])
        for i, chunk_text in enumerate(splits):
            stripped = chunk_text.strip()
            if not stripped:
                continue
            chunks.append({
                "content": stripped,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks (đã có 'embedding') vào Weaviate.

    Tạo (hoặc tái sử dụng) collection `DrugLawDocs` với
    `Configure.Vectorizer.none()` vì vector đã được tính sẵn ở bước
    `embed_chunks` (BAAI/bge-m3) — Weaviate chỉ lưu trữ & lập index ANN.
    """
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.create(
                name=WEAVIATE_COLLECTION,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

        collection = client.collections.get(WEAVIATE_COLLECTION)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                    vector=chunk["embedding"],
                )
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
