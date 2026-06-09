"""
Task 10 — Generation Có Citation.

Pipeline end-to-end:
    1. Retrieve chunks (Task 9)
    2. Reorder để tránh "lost in the middle"
    3. Format context với source labels
    4. Inject vào prompt + call OpenAI
    5. Return answer có citation + sources

=============================================================================
LỰA CHỌN THAM SỐ

top_k = 5:
    Đủ evidence đa dạng mà không làm context quá dài.
    Với văn bản pháp luật tiếng Việt, 5 × 800 chars ≈ 4000 chars — vừa
    với context window của gpt-4o-mini (128k tokens) và không gây noise.

top_p = 0.9 (nucleus sampling):
    Giữ 90% probability mass → đủ đa dạng về từ ngữ mà không quá random.
    Phù hợp cho task factual (pháp luật) cần ngôn ngữ rõ ràng.

temperature = 0.2:
    RAG cần factual accuracy, không cần sáng tạo.
    Thấp hơn 0.3 để câu trả lời nhất quán, ít hallucination hơn.

model = gpt-4o-mini:
    Chi phí thấp, đủ thông minh cho task extraction + citation.
    Nếu cần chất lượng cao hơn: dùng gpt-4o.

=============================================================================
LOST IN THE MIDDLE (Liu et al. 2023):
    LLM chú ý nhiều nhất vào đầu và cuối prompt.
    Context ở giữa bị "bỏ quên" — gọi là "lost in the middle effect".

    Strategy: đặt chunk quan trọng nhất ở vị trí 1 và n,
    kém quan trọng nhất ở giữa.

    Input [1,2,3,4,5] (sorted by score desc) →
    Output [1,3,5,4,2]:
        pos 1 → chunk 1 (best)          ← LLM chú ý nhiều
        pos 2 → chunk 3
        pos 3 → chunk 5 (worst)         ← giữa, ít chú ý
        pos 4 → chunk 4
        pos 5 → chunk 2 (second best)   ← LLM chú ý nhiều
=============================================================================
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env từ root project
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

# Support import trực tiếp khi chạy script
_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K       = 5      # số chunks đưa vào context
TOP_P       = 0.9    # nucleus sampling: 90% probability mass
TEMPERATURE = 0.2    # thấp → factual, ít hallucinate
LLM_MODEL   = "gpt-4o-mini"   # nhanh, rẻ, đủ tốt cho extraction

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về pháp luật ma tuý Việt Nam.
Trả lời câu hỏi hoàn toàn bằng tiếng Việt, dựa CHỈ vào ngữ cảnh được cung cấp.

Quy tắc bắt buộc:
1. Mọi thông tin thực tế PHẢI có trích dẫn ngay sau câu, dạng [Tên nguồn, năm/điều khoản].
   Ví dụ: "Người phạm tội bị phạt tù từ 2-7 năm [Bộ luật Hình sự 2015, Điều 249]"
   hoặc "Ca sĩ Chi Dân bị khởi tố [Tiền Phong, 2024]"
2. Nếu thông tin KHÔNG có trong ngữ cảnh → trả lời:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
3. KHÔNG suy diễn hoặc bổ sung thông tin ngoài ngữ cảnh.
4. Trả lời có cấu trúc rõ ràng, dùng đoạn văn hoặc danh sách khi phù hợp."""


# =============================================================================
# STEP 1 — Document Reordering: tránh lost in the middle
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, dễ quên GIỮA.
    (Liu et al. 2023 — "Lost in the Middle: How Language Models Use Long Contexts")

    Strategy: interleave — best ở đầu, second-best ở cuối, worst ở giữa.

    Input  [1, 2, 3, 4, 5]  (sorted by relevance desc)
    Output [1, 3, 5, 4, 2]

    Args:
        chunks: List sorted by score descending

    Returns:
        Reordered list
    """
    n = len(chunks)
    if n <= 2:
        return chunks

    # Tách thành nhóm lẻ (→ đầu) và chẵn (→ cuối, đảo ngược)
    odd_indexed  = [chunks[i] for i in range(0, n, 2)]     # 0,2,4,... → đầu
    even_indexed = [chunks[i] for i in range(1, n, 2)]     # 1,3,5,... → cuối

    # Đặt odd trước (best ở đầu), even sau đảo ngược (second-best ở cuối)
    return odd_indexed + list(reversed(even_indexed))


# =============================================================================
# STEP 2 — Format context với source labels
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string có source label để LLM cite.

    Mỗi document được đánh số [Document N] để LLM biết cần cite tài liệu nào.
    Source label được đặt rõ để LLM tạo citation chính xác.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta     = chunk.get("metadata", {})
        source   = meta.get("source", f"Tài liệu {i}")
        doc_type = meta.get("type", "unknown")

        # Tạo label thân thiện cho citation
        if doc_type == "legal":
            # legal: lấy tên file bỏ .md
            stem = source.replace(".md", "").replace("-", " ").title()
            label = f"{stem}"
        else:
            # news: lấy tên file (article_01 → bài báo)
            label = source.replace(".md", "").replace("_", " ").title()

        context_parts.append(
            f"[Document {i} | Nguồn: {label} | Loại: {doc_type}]\n"
            f"{chunk['content'].strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# STEP 3 — Call LLM
# =============================================================================

def _call_openai(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI Chat Completions API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY chưa được set.\n"
            "Thêm vào .env: OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature = TEMPERATURE,
        top_p       = TOP_P,
        # max_tokens: không set để LLM tự quyết — tránh cắt giữa câu
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# MAIN: generate_with_citation
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        query
          ↓ Task 9 — retrieve (hybrid + rerank + fallback)
        chunks (top_k)
          ↓ reorder_for_llm — tránh lost in the middle
        reordered
          ↓ format_context — thêm source labels [Document N]
        context_str
          ↓ SYSTEM_PROMPT + context + query → OpenAI
        answer có citation [Nguồn, năm]

    Args:
        query: Câu hỏi
        top_k: Số chunks context (default 5)

    Returns:
        {
            'answer'           : str,        # Câu trả lời có citation
            'sources'          : list[dict], # Chunks đã dùng (original order)
            'reordered_sources': list[dict], # Chunks sau reorder (order trong prompt)
            'retrieval_source' : str,        # 'hybrid' | 'pageindex' | 'none'
            'model'            : str,        # LLM model đã dùng
        }
    """
    # ── Step 1: Retrieve ─────────────────────────────────────────────────────
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer"           : "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources"          : [],
            "reordered_sources": [],
            "retrieval_source" : "none",
            "model"            : LLM_MODEL,
        }

    retrieval_src = chunks[0].get("retrieval_source", "hybrid")

    # ── Step 2: Reorder để tránh lost in the middle ──────────────────────────
    reordered = reorder_for_llm(chunks)

    # ── Step 3: Format context ───────────────────────────────────────────────
    context_str = format_context(reordered)

    # ── Step 4: Build prompt ─────────────────────────────────────────────────
    user_message = (
        f"Ngữ cảnh:\n\n{context_str}\n\n"
        f"{'─' * 60}\n\n"
        f"Câu hỏi: {query}"
    )

    # ── Step 5: Call LLM ─────────────────────────────────────────────────────
    answer = _call_openai(SYSTEM_PROMPT, user_message)

    # ── Step 6: Return ───────────────────────────────────────────────────────
    return {
        "answer"           : answer,
        "sources"          : chunks,
        "reordered_sources": reordered,
        "retrieval_source" : retrieval_src,
        "model"            : LLM_MODEL,
    }


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 10 — RAG Generation với Citation")
    parser.add_argument("query",   nargs="?", default=None, help="Câu hỏi")
    parser.add_argument("--top-k", type=int, default=TOP_K,  help="Số chunks context")
    args = parser.parse_args()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    queries = [args.query] if args.query else test_queries

    for q in queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)

        result = generate_with_citation(q, top_k=args.top_k)

        print(f"\nA:\n{result['answer']}")
        print(f"\n{'─'*70}")
        print(f"[Model: {result['model']} | Via: {result['retrieval_source']} "
              f"| Sources: {len(result['sources'])} chunks]")

        print("\nSources used (original rank):")
        for i, s in enumerate(result["sources"], 1):
            src   = s.get("metadata", {}).get("source", "?")
            typ   = s.get("metadata", {}).get("type", "?")
            score = s.get("score", 0)
            print(f"  [{i}] score={score:+.3f}  [{typ}]  {src}")
