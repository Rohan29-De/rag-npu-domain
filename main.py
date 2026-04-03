"""
main.py
-------
CLI entrypoint for the NPU-domain RAG system.

Usage:
  python main.py ingest                     # build vector store
  python main.py query "Your question"      # single query
  python main.py evaluate                   # run full evaluation
  python main.py evaluate --qualitative     # evaluation + human rubric scoring
  python main.py interactive                # interactive Q&A loop
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure src/ is on the path — must happen before ANY src imports
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def cmd_ingest():
    from ingest import run_ingestion
    run_ingestion()
    print("\n✓ Vector store built. Run 'python main.py query \"...\"' to test.")


def cmd_query(question: str):
    from rag_pipeline import RAGPipeline
    rag    = RAGPipeline()
    result = rag.query(question, verbose=True)
    print(f"\n{'='*70}")
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\n── Retrieved Sources ──")
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['score']:.4f}] {chunk['source']}")
    print(f"\nTokens — prompt: {result['usage']['prompt_tokens']}, "
          f"completion: {result['usage']['completion_tokens']}")


def cmd_evaluate(qualitative: bool = False):
    from rag_pipeline import RAGPipeline
    from evaluator import Evaluator
    rag       = RAGPipeline()
    evaluator = Evaluator(qualitative=qualitative)
    evaluator.run(rag)


def cmd_interactive():
    from rag_pipeline import RAGPipeline
    print("\n🤖 NPU Knowledge Assistant (type 'quit' to exit)\n")
    rag = RAGPipeline()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        result = rag.query(question)
        print(f"\nAssistant: {result['answer']}\n")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "ingest":
        cmd_ingest()

    elif cmd == "query":
        if len(args) < 2:
            print("Usage: python main.py query \"Your question here\"")
            sys.exit(1)
        cmd_query(args[1])

    elif cmd == "evaluate":
        qualitative = "--qualitative" in args
        cmd_evaluate(qualitative)

    elif cmd == "interactive":
        cmd_interactive()

    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()