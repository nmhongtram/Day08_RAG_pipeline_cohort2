# RAG Evaluation Results

**Framework:** RAGAS
**Date:** 2026-06-08 22:38
**Golden Dataset:** 24 Q&A pairs
**Retrieval:** BM25 (rank_bm25) trên group_project\data\standardized
**Generation LLM:** gpt-4o-mini
**Evaluation LLM:** gpt-4o-mini

---

## Overall Scores

| Metric | Config A (top_k=5) | Config B (top_k=2) | Δ (B−A) |
|--------|-------------------|-------------------|---------|
| Faithfulness              | 0.7344 | 0.5003 | -0.2340 |
| Answer Relevancy          | 0.3800 | 0.2667 | -0.1133 |
| Context Recall            | 0.8269 | 0.7269 | -0.1000 |
| Context Precision         | 0.8970 | 0.9375 | +0.0405 |
| **Average                ** | **0.7096** | **0.6079** | **-0.1017** |

---

## A/B Comparison Analysis

**Config A — BM25 retrieval với top_k=5 (nhiều context hơn)**
- top_k = 5 chunks đưa vào LLM context
- Mỗi chunk: size=500, overlap=50
- Ưu điểm: cung cấp nhiều evidence → Context Recall cao hơn
- Nhược điểm: nhiều chunks không liên quan → Context Precision thấp hơn, context dài gây lost-in-the-middle

**Config B — BM25 retrieval với top_k=2 (ít context hơn, ít nhiễu hơn)**
- top_k = 2 chunks đưa vào LLM context
- Mỗi chunk: size=500, overlap=50
- Ưu điểm: context gọn, ít nhiễu → Context Precision cao hơn, LLM ít bị phân tâm
- Nhược điểm: thiếu evidence → Context Recall thấp hơn

**Kết luận:**
Config A (top_k=5) cho kết quả tổng hợp tốt hơn (avg_score: Config A=0.7096 vs Config B=0.6079).
Đây là trade-off điển hình giữa Recall và Precision — cần chọn top_k phù hợp với use case:
- Câu hỏi cần tổng hợp nhiều điều khoản → nên dùng top_k lớn
- Câu hỏi factual đơn giản → top_k nhỏ để tránh nhiễu

---

## Worst Performers (Bottom 3 — Config A)

| # | Question | Avg Score | Root Cause |
|---|----------|-----------|------------|
| 1 | "Tiền chất" theo quy định tại Điều 2 Luật Phòng, chống ma túy 2021 là ... | 0.0812 | BM25 không retrieve đủ evidence (low recall); Context chứa nhiều chunks không liên quan (low precision); LLM hallucinate ngoài context (low faithfulness); Câu trả lời không bám sát câu hỏi (low relevancy) |
| 2 | Theo Điều 4 Nghị định 105/2021/NĐ-CP, "sản xuất chất ma túy" được giải... | 0.1125 | BM25 không retrieve đủ evidence (low recall); Context chứa nhiều chunks không liên quan (low precision); LLM hallucinate ngoài context (low faithfulness); Câu trả lời không bám sát câu hỏi (low relevancy) |
| 3 | Theo Phụ lục Danh mục I của Nghị định 28/2026/NĐ-CP, những chất ma túy... | 0.3750 | LLM hallucinate ngoài context (low faithfulness); Câu trả lời không bám sát câu hỏi (low relevancy) |

**Phân tích:**
Hầu hết các câu hỏi có điểm thấp rơi vào 2 trường hợp:
1. **BM25 không retrieve đúng điều luật** — câu hỏi dùng ngôn ngữ tóm tắt trong khi văn bản pháp luật dùng ngôn ngữ chính thức → keyword mismatch
2. **Context trải rộng nhiều điều khoản** — BM25 trả về chunks từ nhiều chỗ khác nhau trong luật, LLM khó tổng hợp thành câu trả lời mạch lạc

---

## Recommendations

### Cải tiến 1 — Hybrid Search (BM25 + Semantic)
**Action:** Kết hợp BM25 với dense retrieval (BAAI/bge-m3) bằng Reciprocal Rank Fusion (RRF)
**Expected impact:** Context Recall +0.10–0.15 cho các câu hỏi về điều khoản cụ thể; giảm keyword mismatch

### Cải tiến 2 — Cross-encoder Reranking
**Action:** Sau khi lấy top-10 chunks, dùng cross-encoder (jina-reranker-v2-base-multilingual) để rerank và giữ top-5
**Expected impact:** Context Precision +0.08–0.12; loại bỏ chunks không liên quan trước khi đưa vào LLM

### Cải tiến 3 — MarkdownHeader Chunking
**Action:** Thay RecursiveCharacterTextSplitter bằng MarkdownHeaderTextSplitter để giữ nguyên cấu trúc Điều/Khoản
**Expected impact:** Faithfulness +0.05–0.08; mỗi chunk sẽ khép kín một điều luật hoàn chỉnh, LLM dễ cite chính xác hơn
