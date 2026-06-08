"""
RAG Evaluation Pipeline — RAGAS framework.

Implements:
1. SimpleRAGPipeline  — BM25 retrieval + OpenAI generation (self-contained)
2. evaluate_with_ragas — chạy 4 RAGAS metrics
3. compare_configs     — A/B comparison (Config A: top_k=5 vs Config B: top_k=2)
4. export_results      — ghi kết quả ra results.md

Requirements:
    pip install ragas langchain-openai langchain-text-splitters rank-bm25 \
                python-dotenv numpy datasets

Environment (.env):
    OPENAI_API_KEY=sk-...

Usage:
    python group_project/evaluation/eval_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys

# Fix Windows terminal encoding for Vietnamese characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).parent.parent / ".env")  # group_project/.env
load_dotenv()  # fallback: CWD

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
DATA_DIR = Path(__file__).parent.parent / "data" / "standardized"

GENERATION_MODEL = "gpt-4o-mini"
EVAL_LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật phòng, chống ma túy Việt Nam.
Hãy trả lời câu hỏi dựa HOÀN TOÀN vào context được cung cấp dưới đây.
Với mỗi thông tin quan trọng, hãy trích dẫn nguồn trong ngoặc vuông, ví dụ: [Luật 73/2021, Điều 2].
Nếu context không đủ để trả lời, hãy nói: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
Không được suy đoán hoặc thêm thông tin ngoài context."""


# ---------------------------------------------------------------------------
# Config A/B definitions
# ---------------------------------------------------------------------------

CONFIGS = {
    "config_a_full_context": {
        "description": "BM25 retrieval với top_k=5 (nhiều context hơn)",
        "top_k": 5,
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "config_b_minimal_context": {
        "description": "BM25 retrieval với top_k=2 (ít context hơn, ít nhiễu hơn)",
        "top_k": 2,
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
}


# ---------------------------------------------------------------------------
# Simple RAG Pipeline (BM25 + OpenAI)
# ---------------------------------------------------------------------------

class SimpleRAGPipeline:
    """
    BM25-based RAG pipeline dùng cho evaluation.

    Không phụ thuộc vào vector store — chỉ cần rank_bm25 và OpenAI API.
    Documents được load từ group_project/data/standardized/*.md
    """

    def __init__(self, top_k: int = 5, chunk_size: int = 500, chunk_overlap: int = 50):
        self.top_k = top_k
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._build_index(chunk_size, chunk_overlap)

        from openai import OpenAI
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _build_index(self, chunk_size: int, chunk_overlap: int) -> None:
        """Load tất cả .md files từ DATA_DIR và build BM25 index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        md_files = sorted(DATA_DIR.rglob("*.md"))
        if not md_files:
            print(f"  [WARNING] Không tìm thấy .md files trong {DATA_DIR}", file=sys.stderr)

        for md_file in md_files:
            if md_file.name.startswith("."):
                continue
            text = md_file.read_text(encoding="utf-8")
            splits = splitter.split_text(text)
            for i, chunk_text in enumerate(splits):
                if len(chunk_text.strip()) < 30:
                    continue
                self.chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": md_file.stem,
                        "chunk_index": i,
                    },
                })

        tokenized = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Indexed {len(self.chunks)} chunks from {len(md_files)} files")

    def retrieve(self, query: str) -> list[dict]:
        """BM25 retrieval — trả về top_k chunks sorted by score."""
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        return [
            {
                "content": self.chunks[idx]["content"],
                "score": float(scores[idx]),
                "metadata": self.chunks[idx]["metadata"],
            }
            for idx in top_indices
        ]

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Gọi OpenAI để generate câu trả lời dựa trên retrieved chunks."""
        context_parts = [
            f"[Tài liệu {i}: {c['metadata']['source']}]\n{c['content']}"
            for i, c in enumerate(chunks, 1)
        ]
        context = "\n\n---\n\n".join(context_parts)

        response = self.openai.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\n---\nCâu hỏi: {query}"},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def run(self, query: str) -> dict:
        """End-to-end: retrieve → generate → trả về answer + sources."""
        chunks = self.retrieve(query)
        answer = self.generate(query, chunks)
        return {"answer": answer, "sources": chunks}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def collect_pipeline_outputs(pipeline: SimpleRAGPipeline, golden_dataset: list[dict]) -> dict:
    """
    Chạy pipeline trên toàn bộ golden_dataset.

    Returns:
        dict với keys: question, answer, contexts, ground_truth
    """
    eval_data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(golden_dataset):
        print(f"  [{i + 1:02d}/{len(golden_dataset)}] {item['question'][:65]}...")
        result = pipeline.run(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    return eval_data


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def _build_ragas_metrics(llm_wrapper, emb_wrapper):
    """Build RAGAS metric objects (hỗ trợ cả v0.1.x và v0.2.x)."""
    try:
        # RAGAS 0.2.x API
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
        return [
            Faithfulness(llm=llm_wrapper),
            AnswerRelevancy(llm=llm_wrapper, embeddings=emb_wrapper),
            ContextRecall(llm=llm_wrapper),
            ContextPrecision(llm=llm_wrapper),
        ]
    except TypeError:
        # RAGAS 0.1.x — metrics là global singletons
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        faithfulness.llm = llm_wrapper
        answer_relevancy.llm = llm_wrapper
        context_recall.llm = llm_wrapper
        context_precision.llm = llm_wrapper
        return [faithfulness, answer_relevancy, context_recall, context_precision]


def run_ragas_evaluation(eval_data: dict) -> "pd.DataFrame":
    """
    Chạy RAGAS evaluation với 4 metrics.

    Metrics:
        - Faithfulness       : câu trả lời có bám đúng context không?
        - Answer Relevancy   : câu trả lời có đúng câu hỏi không?
        - Context Recall     : retriever có lấy đủ evidence không?
        - Context Precision  : trong context lấy về, bao nhiêu % thực sự hữu ích?
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate

    llm = LangchainLLMWrapper(ChatOpenAI(model=EVAL_LLM_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = _build_ragas_metrics(llm, emb)

    try:
        # RAGAS 0.2.x
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=g,
            )
            for q, a, c, g in zip(
                eval_data["question"],
                eval_data["answer"],
                eval_data["contexts"],
                eval_data["ground_truth"],
            )
        ]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=dataset, metrics=metrics)

    except (ImportError, AttributeError):
        # RAGAS 0.1.x fallback
        from datasets import Dataset

        dataset = Dataset.from_dict(eval_data)
        result = evaluate(dataset=dataset, metrics=metrics)

    return result.to_pandas()


# ---------------------------------------------------------------------------
# A/B Comparison
# ---------------------------------------------------------------------------

def compare_configs(golden_dataset: list[dict]) -> dict[str, dict]:
    """
    Chạy evaluation trên 2 config khác nhau:
        - Config A: top_k=5  (full context, nhiều chunks hơn)
        - Config B: top_k=2  (minimal context, ít chunks hơn)

    Hypothesis:
        Config A → Context Recall cao hơn (nhiều evidence hơn)
        Config B → Context Precision cao hơn (ít noise hơn)
    """
    results: dict[str, dict] = {}

    for config_name, params in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running {config_name}: {params['description']}")
        print(f"{'='*60}")

        pipeline = SimpleRAGPipeline(
            top_k=params["top_k"],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
        )

        print("\nCollecting pipeline outputs...")
        eval_data = collect_pipeline_outputs(pipeline, golden_dataset)

        print("\nRunning RAGAS evaluation...")
        df = run_ragas_evaluation(eval_data)

        results[config_name] = {
            "params": params,
            "eval_data": eval_data,
            "df": df,
            "scores": _aggregate_scores(df),
        }

        print(f"\nScores for {config_name}:")
        for metric, score in results[config_name]["scores"].items():
            print(f"  {metric:25s}: {score:.4f}")

    return results


def _aggregate_scores(df) -> dict[str, float]:
    """Lấy mean score cho mỗi metric từ RAGAS result DataFrame."""
    score_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    return {col: float(df[col].mean()) for col in score_cols}


# ---------------------------------------------------------------------------
# Worst Performers analysis
# ---------------------------------------------------------------------------

def find_worst_performers(df, eval_data: dict, n: int = 3) -> list[dict]:
    """
    Tìm n câu hỏi có điểm thấp nhất (tổng hợp qua các metrics).

    Dùng để phân tích nguyên nhân và đề xuất cải tiến.
    """
    score_cols = [
        c for c in df.columns
        if c not in ("user_input", "response", "retrieved_contexts", "reference")
    ]
    if not score_cols:
        return []

    df = df.copy()
    df["avg_score"] = df[score_cols].mean(axis=1)

    worst_indices = df["avg_score"].nsmallest(n).index.tolist()

    worst = []
    for idx in worst_indices:
        row = df.iloc[idx]
        worst.append({
            "question": eval_data["question"][idx],
            "avg_score": float(row["avg_score"]),
            "scores": {col: float(row[col]) for col in score_cols},
        })
    return worst


def _diagnose_failure(item: dict) -> str:
    """Đơn giản hóa chẩn đoán vì sao một câu hỏi cho điểm thấp."""
    scores = item["scores"]
    reasons = []

    recall_key = next((k for k in scores if "recall" in k.lower()), None)
    precision_key = next((k for k in scores if "precision" in k.lower()), None)
    faithfulness_key = next((k for k in scores if "faithful" in k.lower()), None)
    relevancy_key = next((k for k in scores if "relevance" in k.lower() or "relevancy" in k.lower()), None)

    if recall_key and scores[recall_key] < 0.5:
        reasons.append("BM25 không retrieve đủ evidence (low recall)")
    if precision_key and scores[precision_key] < 0.5:
        reasons.append("Context chứa nhiều chunks không liên quan (low precision)")
    if faithfulness_key and scores[faithfulness_key] < 0.5:
        reasons.append("LLM hallucinate ngoài context (low faithfulness)")
    if relevancy_key and scores[relevancy_key] < 0.5:
        reasons.append("Câu trả lời không bám sát câu hỏi (low relevancy)")

    return "; ".join(reasons) if reasons else "Điểm tổng hợp thấp"


# ---------------------------------------------------------------------------
# Export Results
# ---------------------------------------------------------------------------

def export_results(comparison: dict[str, dict]) -> None:
    """Format evaluation results thành results.md."""

    config_names = list(comparison.keys())
    config_a_name = config_names[0]
    config_b_name = config_names[1]
    config_a = comparison[config_a_name]
    config_b = comparison[config_b_name]

    scores_a = config_a["scores"]
    scores_b = config_b["scores"]

    # Normalize metric names for display
    metric_display = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "response_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    def display(k: str) -> str:
        return metric_display.get(k.lower(), k.replace("_", " ").title())

    all_metric_keys = list(scores_a.keys())

    # Build overall scores table
    rows = []
    for k in all_metric_keys:
        a_val = scores_a.get(k, float("nan"))
        b_val = scores_b.get(k, float("nan"))
        delta = b_val - a_val
        delta_str = f"{delta:+.4f}"
        rows.append(f"| {display(k):25s} | {a_val:.4f} | {b_val:.4f} | {delta_str} |")

    avg_a = float(np.mean(list(scores_a.values())))
    avg_b = float(np.mean(list(scores_b.values())))
    rows.append(f"| **{'Average':23s}** | **{avg_a:.4f}** | **{avg_b:.4f}** | **{avg_b - avg_a:+.4f}** |")

    table = "\n".join(rows)

    # Worst performers
    worst = find_worst_performers(
        config_a["df"], config_a["eval_data"], n=3
    )
    worst_rows = []
    for i, item in enumerate(worst, 1):
        q_short = item["question"][:70] + ("..." if len(item["question"]) > 70 else "")
        scores_str = " / ".join(f"{display(k)}={v:.2f}" for k, v in item["scores"].items())
        root_cause = _diagnose_failure(item)
        worst_rows.append(
            f"| {i} | {q_short} | {item['avg_score']:.4f} | {root_cause} |"
        )

    worst_table = "\n".join(worst_rows) if worst_rows else "| - | - | - | - |"

    content = f"""# RAG Evaluation Results

**Framework:** RAGAS
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Golden Dataset:** {len(config_a["eval_data"]["question"])} Q&A pairs
**Retrieval:** BM25 (rank_bm25) trên {DATA_DIR.relative_to(Path(__file__).parent.parent.parent) if DATA_DIR.is_relative_to(Path(__file__).parent.parent.parent) else DATA_DIR}
**Generation LLM:** {GENERATION_MODEL}
**Evaluation LLM:** {EVAL_LLM_MODEL}

---

## Overall Scores

| Metric | Config A (top_k=5) | Config B (top_k=2) | Δ (B−A) |
|--------|-------------------|-------------------|---------|
{table}

---

## A/B Comparison Analysis

**Config A — {config_a["params"]["description"]}**
- top_k = {config_a["params"]["top_k"]} chunks đưa vào LLM context
- Mỗi chunk: size={config_a["params"]["chunk_size"]}, overlap={config_a["params"]["chunk_overlap"]}
- Ưu điểm: cung cấp nhiều evidence → Context Recall cao hơn
- Nhược điểm: nhiều chunks không liên quan → Context Precision thấp hơn, context dài gây lost-in-the-middle

**Config B — {config_b["params"]["description"]}**
- top_k = {config_b["params"]["top_k"]} chunks đưa vào LLM context
- Mỗi chunk: size={config_b["params"]["chunk_size"]}, overlap={config_b["params"]["chunk_overlap"]}
- Ưu điểm: context gọn, ít nhiễu → Context Precision cao hơn, LLM ít bị phân tâm
- Nhược điểm: thiếu evidence → Context Recall thấp hơn

**Kết luận:**
{"Config A (top_k=5) cho kết quả tổng hợp tốt hơn" if avg_a >= avg_b else "Config B (top_k=2) cho kết quả tổng hợp tốt hơn"} (avg_score: Config A={avg_a:.4f} vs Config B={avg_b:.4f}).
Đây là trade-off điển hình giữa Recall và Precision — cần chọn top_k phù hợp với use case:
- Câu hỏi cần tổng hợp nhiều điều khoản → nên dùng top_k lớn
- Câu hỏi factual đơn giản → top_k nhỏ để tránh nhiễu

---

## Worst Performers (Bottom 3 — Config A)

| # | Question | Avg Score | Root Cause |
|---|----------|-----------|------------|
{worst_table}

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
"""

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n  Results exported to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY không được set. Thêm vào file .env rồi chạy lại.", file=sys.stderr)
        sys.exit(1)

    print("Loading golden dataset...")
    golden_dataset = load_golden_dataset()
    print(f"  Loaded {len(golden_dataset)} test cases")

    print("\nRunning A/B comparison (Config A: top_k=5 vs Config B: top_k=2)...")
    comparison = compare_configs(golden_dataset)

    print("\nExporting results to results.md...")
    export_results(comparison)

    print("\nDone! Xem kết quả tại group_project/evaluation/results.md")


if __name__ == "__main__":
    main()
