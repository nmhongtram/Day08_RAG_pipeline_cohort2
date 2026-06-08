"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve   # ⚠ đổi sang absolute import nếu chạy trực tiếp


# =============================================================================
# CONFIG
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source.

If the information is not explicitly stated in the provided context,
state 'Tôi không thể xác minh thông tin này từ nguồn hiện có'.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- No hallucination
- Structure clearly"""


# =============================================================================
# REORDER
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks

    odd = [chunks[i] for i in range(0, len(chunks), 2)]   # 1,3,5
    even = [chunks[i] for i in range(1, len(chunks), 2)]  # 2,4

    return odd + even[::-1]


# =============================================================================
# FORMAT CONTEXT
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")

        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content'].strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Thiếu OPENAI_API_KEY")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none"
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Prompt
    user_message = f"""
Context:
{context}

---

Question: {query}
"""

    # Step 5: Call LLM
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=800
    )

    answer = response.choices[0].message.content.strip()

    # Step 6: Safety check (nếu model không cite)
    if "[" not in answer:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid")
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
        "Nghệ sĩ nào bị bắt vì ma tuý?",
        "Quy trình cai nghiện bắt buộc là gì?",
    ]

    for q in queries:
        print("\n" + "="*60)
        print("Q:", q)

        try:
            result = generate_with_citation(q)
            print("\nA:", result["answer"])

            print(f"\nSources ({len(result['sources'])}):")
            for i, s in enumerate(result["sources"], 1):
                print(f"  {i}. score={s.get('score', 0):.3f}")

        except Exception as e:
            print("Lỗi:", e)