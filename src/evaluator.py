"""
evaluator.py
------------
Comprehensive RAG evaluation framework with:
  1. Quantitative Metrics
     - ROUGE-1 / ROUGE-L F1      (n-gram overlap)
     - Semantic Similarity Score  (cosine similarity of answer embeddings)
     - Keyword Hit Rate           (domain keyword presence)
     - Retrieval Relevance Score  (avg cosine score of retrieved chunks)
  2. Retrieval Performance
     - Source Hit Rate            (did we retrieve the correct source doc?)
  3. Qualitative Assessment
     - Structured human-scoring rubric displayed per question
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure src/ is on path for sibling imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
QA_PATH  = ROOT / "qa_pairs.json"
RESULTS_DIR = ROOT / "eval_results"
# ──────────────────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"

# Domain-specific keywords to test for in answers
DOMAIN_KEYWORDS = [
    "NPU", "neural", "systolic", "TOPS", "inference", "training",
    "embedding", "quantization", "SRAM", "MAC", "accelerator",
    "TPU", "matrix", "bandwidth", "transformer"
]


# ── Metric Functions ───────────────────────────────────────────────────────────

def compute_rouge(generated: str, reference: str) -> dict[str, float]:
    """ROUGE-1 and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1_f1" : round(scores["rouge1"].fmeasure, 4),
        "rougeL_f1" : round(scores["rougeL"].fmeasure, 4),
    }


def compute_semantic_similarity(generated: str, reference: str,
                                 model: SentenceTransformer) -> float:
    """Cosine similarity between generated and reference answer embeddings."""
    vecs = model.encode([generated, reference], normalize_embeddings=True,
                        convert_to_numpy=True)
    sim  = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
    return round(sim, 4)


def compute_keyword_hit_rate(answer: str,
                              keywords: list[str] = DOMAIN_KEYWORDS) -> float:
    """Fraction of domain keywords present in the generated answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


def compute_retrieval_relevance(chunks: list[dict]) -> float:
    """Mean cosine similarity score of the retrieved chunks."""
    if not chunks:
        return 0.0
    return round(float(np.mean([c["score"] for c in chunks])), 4)


def check_source_hit(chunks: list[dict], expected_source: str) -> bool:
    """Did the retriever return at least one chunk from the expected source doc?"""
    retrieved_sources = {c["source"].replace(".txt", "") for c in chunks}
    return expected_source in retrieved_sources


# ── Qualitative Rubric Display ─────────────────────────────────────────────────

RUBRIC = {
    "Coherence"           : "Is the answer grammatically correct and logically structured? (1=poor, 5=excellent)",
    "Completeness"        : "Does the answer cover all key aspects of the question? (1=missing most, 5=comprehensive)",
    "Factual Correctness" : "Are the facts in the answer accurate per the reference? (1=wrong, 5=fully correct)",
    "Conciseness"         : "Is the answer appropriately concise without losing substance? (1=bloated, 5=tight)",
}


def display_qualitative_rubric(result: dict, expected_answer: str) -> dict[str, int]:
    """
    Print a comparison and prompt human scorer for rubric scores.
    Returns dict of criterion → score. In non-interactive mode returns empty dict.
    """
    print("\n" + "─"*70)
    print(f"  Q: {result['question']}")
    print(f"\n  GENERATED:\n  {result['answer']}")
    print(f"\n  EXPECTED:\n  {expected_answer}")
    print("\n  ── QUALITATIVE RUBRIC SCORING ──")
    scores = {}
    for criterion, description in RUBRIC.items():
        while True:
            try:
                raw = input(f"  {criterion} — {description}\n  Score [1-5]: ").strip()
                score = int(raw)
                if 1 <= score <= 5:
                    scores[criterion] = score
                    break
                print("  Please enter a number between 1 and 5.")
            except (ValueError, EOFError):
                print("  Skipping (non-interactive mode).")
                scores[criterion] = None
                break
    return scores


# ── Main Evaluation Runner ─────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, qualitative: bool = False):
        self.qualitative = qualitative
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.qa_pairs    = json.loads(QA_PATH.read_text())
        RESULTS_DIR.mkdir(exist_ok=True)

    def evaluate_single(self, result: dict, qa_item: dict) -> dict:
        """Compute all metrics for one Q&A result."""
        generated = result["answer"]
        reference = qa_item["expected_answer"]
        chunks    = result["retrieved_chunks"]

        rouge     = compute_rouge(generated, reference)
        sem_sim   = compute_semantic_similarity(generated, reference, self.embed_model)
        kw_rate   = compute_keyword_hit_rate(generated)
        ret_rel   = compute_retrieval_relevance(chunks)
        src_hit   = check_source_hit(chunks, qa_item["source_doc"])

        metrics = {
            "question_id"          : qa_item["id"],
            "question"             : qa_item["question"],
            "rouge1_f1"            : rouge["rouge1_f1"],
            "rougeL_f1"            : rouge["rougeL_f1"],
            "semantic_similarity"  : sem_sim,
            "keyword_hit_rate"     : kw_rate,
            "retrieval_relevance"  : ret_rel,
            "source_hit"           : src_hit,
            "generated_answer"     : generated,
            "expected_answer"      : reference,
        }

        # Composite score: weighted average of core metrics
        metrics["composite_score"] = round(
            0.30 * sem_sim +
            0.25 * rouge["rouge1_f1"] +
            0.20 * kw_rate +
            0.15 * ret_rel +
            0.10 * float(src_hit),
            4
        )

        if self.qualitative:
            qual = display_qualitative_rubric(result, reference)
            metrics["qualitative_scores"] = qual
            if any(v is not None for v in qual.values()):
                valid = [v for v in qual.values() if v is not None]
                metrics["qualitative_avg"] = round(sum(valid) / len(valid), 2)

        return metrics

    def run(self, rag_pipeline) -> pd.DataFrame:
        """
        Run evaluation over all QA pairs.
        Returns a DataFrame of per-question metrics.
        """
        all_metrics = []

        print(f"\n{'='*70}")
        print(f"  RAG EVALUATION — {len(self.qa_pairs)} questions")
        print(f"  Qualitative mode: {self.qualitative}")
        print(f"{'='*70}\n")

        for i, qa in enumerate(self.qa_pairs, 1):
            print(f"[{i:02d}/{len(self.qa_pairs)}] Querying: {qa['question'][:60]}…")
            result  = rag_pipeline.query(qa["question"])
            metrics = self.evaluate_single(result, qa)
            all_metrics.append(metrics)
            print(f"       → Semantic Sim: {metrics['semantic_similarity']:.4f} | "
                  f"ROUGE-1: {metrics['rouge1_f1']:.4f} | "
                  f"Composite: {metrics['composite_score']:.4f} | "
                  f"SrcHit: {metrics['source_hit']}")

        df = pd.DataFrame(all_metrics)
        self._print_summary(df)
        self._save_results(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        print(f"\n{'='*70}")
        print("  EVALUATION SUMMARY")
        print(f"{'='*70}")
        numeric_cols = ["rouge1_f1", "rougeL_f1", "semantic_similarity",
                        "keyword_hit_rate", "retrieval_relevance", "composite_score"]
        for col in numeric_cols:
            print(f"  {col:<28} mean={df[col].mean():.4f}  "
                  f"min={df[col].min():.4f}  max={df[col].max():.4f}")
        src_hit_rate = df["source_hit"].mean()
        print(f"  {'source_hit_rate':<28} {src_hit_rate:.2%}")
        print(f"\n  Best Q  (composite): {df.loc[df['composite_score'].idxmax(), 'question'][:60]}")
        print(f"  Worst Q (composite): {df.loc[df['composite_score'].idxmin(), 'question'][:60]}")
        print(f"{'='*70}\n")

    def _save_results(self, df: pd.DataFrame) -> None:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv = RESULTS_DIR / f"eval_{ts}.csv"
        df.to_csv(csv, index=False)
        print(f"[evaluator] Results saved → {csv}")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from rag_pipeline import RAGPipeline

    qualitative = "--qualitative" in sys.argv
    rag         = RAGPipeline()
    evaluator   = Evaluator(qualitative=qualitative)
    df          = evaluator.run(rag)

    print("\nTop 5 questions by composite score:")
    print(df[["question_id", "composite_score", "semantic_similarity", "source_hit"]]
          .sort_values("composite_score", ascending=False).head(5).to_string(index=False))