"""
Shared embedding helper — BAAI/bge-m3.

Lý do chọn BAAI/bge-m3:
    - Multilingual, hỗ trợ tốt tiếng Việt (corpus của chúng ta toàn văn bản
      pháp luật + tin tức tiếng Việt)
    - 1024 chiều — đủ biểu đạt ngữ nghĩa cho câu dài (luật có nhiều câu phức)
    - Cùng 1 model dùng xuyên suốt: SemanticChunker (Task 4), semantic_search
      (Task 5) và MMR rerank (Task 7) — đảm bảo các vector nằm cùng không gian.

Module này cung cấp:
    - get_embedding_model(): trả về SentenceTransformer singleton (load 1 lần)
    - embed_texts(texts): encode list[str] -> list[list[float]]
    - LangchainEmbeddings: wrapper tương thích interface `Embeddings` của
      LangChain (embed_documents/embed_query) — dùng BGE-M3, cho mọi nơi cần
      vector "thật" (lưu trữ + truy vấn: Task 4 embed_chunks, Task 5, Task 7 MMR)
    - FastBreakpointEmbeddings: wrapper nhẹ (all-MiniLM-L6-v2) CHỈ dùng nội bộ
      bởi SemanticChunker để dò ranh giới ngữ nghĩa (xem ghi chú bên dưới)

Ghi chú hiệu năng — vì sao SemanticChunker dùng model phụ để dò breakpoint:
    SemanticChunker phải embed TỪNG câu trong văn bản để tính similarity giữa
    các câu liền kề rồi tìm điểm "gãy" ngữ nghĩa. BGE-M3 (~568M tham số) chạy
    trên CPU cho hàng trăm câu mất nhiều phút/tài liệu — không khả thi để chạy
    toàn bộ corpus hay test suite. all-MiniLM-L6-v2 (22M tham số, 384-dim) cho
    kết quả dò breakpoint tương đương (ta chỉ cần THỨ TỰ tương đối của độ
    tương đồng giữa các câu, không cần giá trị embedding tuyệt đối) nhưng
    nhanh hơn ~15-20 lần. BGE-M3 vẫn là embedding model DUY NHẤT dùng để lưu
    trữ & truy vấn (EMBEDDING_MODEL/EMBEDDING_DIM = 1024) — đúng yêu cầu đề bài.
"""

from functools import lru_cache

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

_BREAKPOINT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load (và cache) SentenceTransformer cho BAAI/bge-m3."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_breakpoint_model():
    """Load (và cache) model nhẹ dùng riêng để dò breakpoint trong SemanticChunker."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_BREAKPOINT_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một danh sách văn bản bằng BGE-M3, trả về list of vectors (1024-dim)."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed một câu truy vấn đơn lẻ bằng BGE-M3."""
    return embed_texts([query])[0]


class LangchainEmbeddings:
    """
    Wrapper mỏng quanh `get_embedding_model()` (BGE-M3) để khớp interface
    `langchain_core.embeddings.Embeddings` (embed_documents/embed_query).
    Dùng cho mọi nơi cần vector lưu trữ/truy vấn thật sự.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)


class FastBreakpointEmbeddings:
    """
    Wrapper nhẹ (all-MiniLM-L6-v2) CHỈ dùng làm "thước đo tương đồng nội bộ"
    cho SemanticChunker khi dò breakpoint giữa các câu — xem ghi chú hiệu
    năng ở đầu file. KHÔNG dùng để lưu trữ hay truy vấn (vector cuối cùng
    luôn là BGE-M3, qua `LangchainEmbeddings`/`embed_texts`).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_breakpoint_model()
        embeddings = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
