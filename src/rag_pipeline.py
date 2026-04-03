"""
rag_pipeline.py
---------------
Combines the Retriever with the Groq LLM to produce grounded answers.
Uses a structured prompt that instructs the model to answer only from
the retrieved context and cite its sources.
"""

import os
import sys
from pathlib import Path

# Ensure src/ is on path for sibling imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from groq import Groq
from retriever import Retriever

GROQ_MODEL  = "llama-3.3-70b-versatile"
TOP_K       = 5
MAX_TOKENS  = 512


SYSTEM_PROMPT = """You are an expert AI hardware analyst specializing in Neural Processing Units (NPUs).
You will be given a user question and a set of retrieved context passages from a specialized NPU knowledge base.

Your task:
1. Answer the question accurately and concisely using ONLY the provided context.
2. If the context does not contain enough information, say so clearly.
3. At the end of your answer, list the source documents you used in a "Sources:" line.
4. Do not hallucinate facts not present in the context.
5. Keep answers focused and under 250 words unless the question requires more detail."""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Context {i} | Source: {chunk['source']} | Score: {chunk['score']:.4f}]")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return f"""CONTEXT:
{context}

QUESTION: {question}

Please answer based solely on the context above."""


class RAGPipeline:
    def __init__(self, top_k: int = TOP_K):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file and run: "
                "source .env  OR  export GROQ_API_KEY=your_key"
            )
        self.client    = Groq(api_key=api_key)
        self.retriever = Retriever()
        self.top_k     = top_k

    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build prompt
          3. Call Groq LLM
          4. Return structured result dict
        """
        # Step 1 — Retrieve
        chunks = self.retriever.retrieve(question, top_k=self.top_k)

        if verbose:
            print(f"\n[RAG] Retrieved {len(chunks)} chunks for: '{question}'")
            for c in chunks:
                print(f"  • [{c['score']:.4f}] {c['chunk_id']}")

        # Step 2 — Build prompt
        user_prompt = build_user_prompt(question, chunks)

        # Step 3 — LLM call
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.2,      # low temp for factual grounding
        )

        answer = response.choices[0].message.content.strip()

        return {
            "question"        : question,
            "answer"          : answer,
            "retrieved_chunks": chunks,
            "model"           : GROQ_MODEL,
            "usage"           : {
                "prompt_tokens"    : response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag = RAGPipeline()

    test_questions = [
        "What is a systolic array and how does it work in an NPU?",
        "Who coined the term neuromorphic engineering?",
        "What makes the Cerebras Wafer Scale Engine unique?",
    ]

    for q in test_questions:
        result = rag.query(q, verbose=True)
        print(f"\n{'='*70}")
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nTokens used: {result['usage']}")