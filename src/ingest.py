"""
ingest.py
---------
Loads documents from the data/ folder, splits them into overlapping chunks,
generates sentence-transformer embeddings, and persists a FAISS index + metadata.
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent.parent / "data"
INDEX_PATH    = Path(__file__).parent.parent / "vector_store" / "faiss.index"
META_PATH     = Path(__file__).parent.parent / "vector_store" / "metadata.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
CHUNK_SIZE    = 400                    # characters per chunk
CHUNK_OVERLAP = 80                     # overlap between consecutive chunks
# ──────────────────────────────────────────────────────────────────────────────


def load_documents(data_dir: Path) -> list[dict]:
    """Read all .txt files from data_dir and return list of {source, text} dicts."""
    docs = []
    for fp in sorted(data_dir.glob("*.txt")):
        text = fp.read_text(encoding="utf-8").strip()
        docs.append({"source": fp.name, "text": text})
    print(f"[ingest] Loaded {len(docs)} documents from {data_dir}")
    return docs


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split document text into overlapping character-level chunks.
    Each chunk inherits the source filename and gets a chunk_id.
    """
    text   = doc["text"]
    source = doc["source"]
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id" : f"{source}__chunk{idx}",
                "source"   : source,
                "text"     : chunk,
                "start_pos": start,
                "end_pos"  : end,
            })
        start += chunk_size - overlap
        idx   += 1

    return chunks


def embed_chunks(chunks: list[dict], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Encode chunk texts into a float32 embedding matrix."""
    model  = SentenceTransformer(model_name)
    texts  = [c["text"] for c in chunks]
    print(f"[ingest] Encoding {len(texts)} chunks with '{model_name}'…")
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index (cosine similarity via L2-normalised vecs)."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)        # Inner Product == cosine for unit vecs
    index.add(embeddings)
    print(f"[ingest] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_artifacts(index: faiss.Index, chunks: list[dict],
                   index_path: Path, meta_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[ingest] Saved index → {index_path}")
    print(f"[ingest] Saved metadata → {meta_path}")


def run_ingestion() -> tuple[faiss.Index, list[dict]]:
    """Full ingestion pipeline. Returns (index, chunks) for in-process use."""
    docs   = load_documents(DATA_DIR)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    print(f"[ingest] Total chunks after splitting: {len(chunks)}")

    embeddings = embed_chunks(chunks)
    index      = build_faiss_index(embeddings)
    save_artifacts(index, chunks, INDEX_PATH, META_PATH)
    return index, chunks


if __name__ == "__main__":
    run_ingestion()
    print("[ingest] ✓ Ingestion complete.")