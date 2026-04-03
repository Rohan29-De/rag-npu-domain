"""
retriever.py
------------
Loads the persisted FAISS index and metadata, then retrieves the top-k most
relevant chunks for a given query using cosine similarity.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH  = Path(__file__).parent.parent / "vector_store" / "faiss.index"
META_PATH   = Path(__file__).parent.parent / "vector_store" / "metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH,
                 meta_path: Path = META_PATH,
                 model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)
        print(f"[retriever] Loaded index with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Encode the query, search FAISS, and return top_k chunk dicts with scores.
        Each returned dict has keys: chunk_id, source, text, score.
        """
        q_vec = self.model.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])   # shallow copy
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def retrieve_with_dedup(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k*2 candidates and deduplicate by source document,
        keeping the highest-scoring chunk per document before trimming to top_k.
        Useful when a document has many overlapping chunks.
        """
        candidates = self.retrieve(query, top_k=top_k * 2)
        seen_sources: set[str] = set()
        deduped: list[dict]    = []
        for chunk in candidates:
            if chunk["source"] not in seen_sources:
                deduped.append(chunk)
                seen_sources.add(chunk["source"])
            if len(deduped) >= top_k:
                break
        return deduped


if __name__ == "__main__":
    r = Retriever()
    results = r.retrieve("What is a systolic array?", top_k=3)
    for res in results:
        print(f"\n[{res['score']:.4f}] {res['source']} — {res['chunk_id']}")
        print(res["text"][:200], "…")