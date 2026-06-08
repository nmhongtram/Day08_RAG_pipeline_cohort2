"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store

=============================================================================
LỰA CHỌN VÀ GIẢI THÍCH
=============================================================================

Chunking: RecursiveCharacterTextSplitter
    - Lý do: Phù hợp nhất cho văn bản hỗn hợp (pháp luật + báo). Tách theo
      thứ tự ưu tiên "\n\n" → "\n" → ". " → " " nên giữ nguyên ngữ cảnh
      đoạn văn tốt hơn fixed-size split đơn giản.
    - chunk_size=800: Đủ lớn để giữ ngữ cảnh một điều luật / đoạn báo (văn
      bản tiếng Việt tốn ký tự hơn), không quá lớn để embedding chính xác.
    - chunk_overlap=100: ~12% overlap giúp câu hỏi không bị "rơi vào khe"
      giữa 2 chunk liền kề.

Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
    - Lý do: Nhẹ (80MB), inference nhanh trên CPU, không cần GPU. Đủ tốt cho
      tiếng Việt ở mức demo. BAAI/bge-m3 tốt hơn nhưng cần >1GB RAM và chậm
      hơn nhiều. Có thể nâng cấp lên bge-m3 sau khi pipeline hoạt động ổn.

Vector Store: ChromaDB (persistent, local)
    - Lý do: Không cần Docker/server, lưu xuống disk, tích hợp đơn giản.
      Đủ tốt cho demo và các task tiếp theo (task 5, 6, 9).
    - Weaviate mạnh hơn (hybrid search built-in) nhưng cần Docker setup.

=============================================================================
"""

import json
import uuid
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORDB_DIR     = Path(__file__).parent.parent / "data" / "vectordb"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking
CHUNK_SIZE    = 800   # chars — giữ đủ ngữ cảnh một điều luật / đoạn báo
CHUNK_OVERLAP = 100   # ~12% overlap để câu hỏi không rơi vào khe giữa chunks

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384  # dimension của all-MiniLM-L6-v2

# Vector store
COLLECTION_NAME = "DrugLawDocs"


# =============================================================================
# STEP 1 — Load documents
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str,
                                               'filepath': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue  # bỏ qua file rỗng (nghi-dinh-105-2021.md ~0KB)

        # Xác định loại tài liệu từ đường dẫn
        relative = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in str(relative) else "news"

        documents.append({
            "content": content,
            "metadata": {
                "source":   md_file.name,
                "type":     doc_type,
                "filepath": str(relative),
            }
        })

    return documents


# =============================================================================
# STEP 2 — Chunk documents
# =============================================================================

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Lý do chọn separators:
        - "\\n\\n": tách theo đoạn văn (heading, paragraph)
        - "\\n"   : tách theo dòng
        - ". "    : tách theo câu (giữ dấu chấm ở đầu chunk sau)
        - " "     : tách theo từ (last resort)
        - ""      : tách theo ký tự (absolute last resort)

    Returns:
        List of {'content': str, 'metadata': dict}  — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # RecursiveCharacterTextSplitter tách theo thứ tự ưu tiên separator list
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:  # bỏ qua chunk quá ngắn (header, ký tự lẻ)
                continue
            chunks.append({
                "content":  chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_total": len(splits),
                }
            })

    return chunks


# =============================================================================
# STEP 3 — Embed chunks
# =============================================================================

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng all-MiniLM-L6-v2 (384 dim).

    Lý do dùng batch_size=64: đủ để tận dụng CPU parallelism mà không OOM.
    normalize_embeddings=True: cosine similarity = dot product → tương thích
    với ChromaDB cosine distance.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (batch_size=64) ...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity = dot product
        convert_to_numpy=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


# =============================================================================
# STEP 4 — Index vào ChromaDB
# =============================================================================

def index_to_vectorstore(chunks: list[dict]) -> None:
    """
    Lưu chunks vào ChromaDB (persistent local, không cần Docker).

    Collection settings:
        - cosine distance (phù hợp với normalized embeddings)
        - upsert thay insert để idempotent khi chạy lại
    """
    import chromadb
    from chromadb.config import Settings

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Connecting to ChromaDB at: {VECTORDB_DIR}")
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Xoá collection cũ nếu tồn tại (idempotent re-index)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Dropping existing collection '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Batch upsert (ChromaDB recommend ≤5000 per call)
    BATCH = 500
    total = len(chunks)
    for start in range(0, total, BATCH):
        batch = chunks[start : start + BATCH]
        collection.add(
            ids         =[str(uuid.uuid4()) for _ in batch],
            embeddings  =[c["embedding"] for c in batch],
            documents   =[c["content"]   for c in batch],
            metadatas   =[c["metadata"]  for c in batch],
        )
        end = min(start + BATCH, total)
        print(f"  Indexed {end}/{total} chunks ...")

    # Verify
    count = collection.count()
    print(f"  ✓ Collection '{COLLECTION_NAME}' has {count} vectors in ChromaDB")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline() -> None:
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking  : RecursiveCharacterTextSplitter")
    print(f"              chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  Embedding : {EMBEDDING_MODEL}  (dim={EMBEDDING_DIM})")
    print(f"  VectorDB  : ChromaDB (persistent local)")
    print("=" * 60)

    # --- Load ---
    print("\n[1/4] Loading documents ...")
    docs = load_documents()
    print(f"  ✓ Loaded {len(docs)} documents")
    for d in docs:
        chars = len(d["content"])
        print(f"     {d['metadata']['filepath']:55s}  {chars:6,} chars")

    # --- Chunk ---
    print(f"\n[2/4] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_documents(docs)
    print(f"  ✓ Created {len(chunks)} chunks")
    legal_n = sum(1 for c in chunks if c["metadata"]["type"] == "legal")
    news_n  = sum(1 for c in chunks if c["metadata"]["type"] == "news")
    print(f"     legal: {legal_n}  |  news: {news_n}")

    # --- Embed ---
    print(f"\n[3/4] Embedding ...")
    chunks = embed_chunks(chunks)
    print(f"  ✓ Embedded {len(chunks)} chunks  (dim={EMBEDDING_DIM})")

    # --- Index ---
    print(f"\n[4/4] Indexing to ChromaDB ...")
    index_to_vectorstore(chunks)

    print("\n" + "=" * 60)
    print(f"✓ Task 4 hoàn thành!")
    print(f"  Tổng chunks : {len(chunks)}")
    print(f"  VectorDB    : {VECTORDB_DIR}")
    print(f"  Collection  : {COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
