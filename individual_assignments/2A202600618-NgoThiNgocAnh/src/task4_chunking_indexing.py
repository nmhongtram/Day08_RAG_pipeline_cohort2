"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB — local, không cần server)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

from pathlib import Path
import json

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter
#   - Lý do chọn "recursive": phù hợp với văn bản pháp luật & báo tiếng Việt
#     vì tách theo thứ tự ưu tiên \n\n → \n → dấu chấm → khoảng trắng,
#     giữ nguyên đoạn văn trọn nghĩa thay vì cắt giữa câu.
#   - CHUNK_SIZE=500: đủ chứa 1 điều luật hoặc 1-2 đoạn báo, không quá dài
#     làm loãng embedding, không quá ngắn làm mất ngữ cảnh.
#   - CHUNK_OVERLAP=50: giữ lại ~1 câu giữa 2 chunk liền kề để truy vấn
#     không bị mất thông tin ở ranh giới chunk.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: BAAI/bge-m3
#   - Lý do: model multilingual tốt nhất cho tiếng Việt trong nhóm open-source,
#     hỗ trợ dense + sparse + colbert (hybrid search), dim=1024 đủ biểu diễn
#     văn bản pháp luật có từ chuyên ngành.
#   - Thay thế nhẹ hơn: "sentence-transformers/all-MiniLM-L6-v2" (384 dim)
#     nếu máy yếu hoặc cần tốc độ index nhanh hơn.
# EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 1024

# Vector Store: ChromaDB
#   - Lý do: chạy local không cần server, phù hợp cho assignment/demo,
#     hỗ trợ metadata filtering và cosine similarity sẵn có.
#   - Nếu cần production/hybrid search: chuyển sang Weaviate (xem phần comment).
VECTOR_STORE = "chromadb"       # "chromadb" | "weaviate"
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "data" / "vectorstore")
COLLECTION_NAME = "drug_law_docs"


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

    if not STANDARDIZED_DIR.exists():
        print(f"  ⚠ Thư mục không tồn tại: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
                continue
            doc_type = "legal" if "legal" in str(md_file) else "news"
            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(md_file),
                }
            })
        except Exception as e:
            print(f"  ✗ Không đọc được {md_file.name}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )

    chunks = []

    if CHUNKING_METHOD == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_index": i},
                    })

    elif CHUNKING_METHOD == "markdown_header":
        # Tách theo heading Markdown — tốt khi file có cấu trúc # / ## / ###
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        # Sau khi tách theo header, tiếp tục tách các đoạn quá dài
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for doc in documents:
            header_chunks = header_splitter.split_text(doc["content"])
            for hc in header_chunks:
                sub_splits = char_splitter.split_text(hc.page_content)
                for i, chunk_text in enumerate(sub_splits):
                    chunk_text = chunk_text.strip()
                    if chunk_text:
                        chunks.append({
                            "content": chunk_text,
                            "metadata": {
                                **doc["metadata"],
                                **hc.metadata,
                                "chunk_index": i,
                            },
                        })

    else:
        raise ValueError(f"Chunking method không hỗ trợ: {CHUNKING_METHOD}")

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,          # tăng nếu có GPU, giảm nếu RAM thấp
        normalize_embeddings=True,  # cosine similarity = dot product sau normalize
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks

def index_to_vectorstore(chunks: list[dict]):
    import chromadb
    from chromadb.config import Settings

    CHROMA_DB_PATH = str(Path(__file__).parent.parent / "data" / "vectorstore")
    COLLECTION_NAME = "drug_law_docs"

    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  ↺ Đã xoá collection cũ: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  → Indexed batch {i // BATCH_SIZE + 1}: {i + len(batch)}/{len(chunks)} chunks")

    print(f"\n  ✓ Collection '{COLLECTION_NAME}' tại: {CHROMA_DB_PATH}")
    print(f"  ✓ Tổng số chunks đã index: {collection.count()}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Cache embeddings để không phải re-embed nếu chạy lại
    cache_path = Path(__file__).parent.parent / "data" / "chunks_cache.json"
    if cache_path.exists():
        print("✓ Dùng embeddings từ cache (bỏ qua bước embed)")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        for chunk, c in zip(chunks, cached):
            chunk["embedding"] = c["embedding"]
    else:
        chunks = embed_chunks(chunks)
        print(f"✓ Embedded {len(chunks)} chunks")
        cache_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Đã lưu cache tại: {cache_path}")

    index_to_vectorstore(chunks)
    print("\n✓ Pipeline hoàn tất!")


if __name__ == "__main__":
    run_pipeline()