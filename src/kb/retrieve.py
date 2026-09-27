"""
Module: src/kb/retrieve.py
Description: Vector similarity retrieval module over the metallurgical ChromaDB index.
Inputs: query (str), optional k (int, default=3), optional defect_class_filter (str)
Outputs: list of dicts with {"text": str, "source": str, "score": float, "id": str, "defect_class": str}
# OWNER: Ravi (built by Aman for now)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "metallurgical_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons for performance
_CLIENT = None
_COLLECTION = None
_EMBEDDER = None


def _get_resources():
    """Lazily load and cache ChromaDB client and embedding model."""
    global _CLIENT, _COLLECTION, _EMBEDDER
    if _CLIENT is None:
        if not CHROMA_DIR.exists():
            # If index not yet built, build it now
            from src.kb.build_index import build_index
            build_index()

        _CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = _CLIENT.get_collection(name=COLLECTION_NAME)
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _COLLECTION, _EMBEDDER


def retrieve(
    query: str,
    k: int = 3,
    defect_class_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant metallurgical passages for a given natural language query.

    Args:
        query: Query string describing defect symptoms, morphology, and context.
        k: Number of passages to retrieve.
        defect_class_filter: Optional defect class name to narrow search.

    Returns:
        List of dicts: [{"text": str, "source": str, "score": float, "id": str, "defect_class": str}]
    """
    collection, embedder = _get_resources()

    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

    where_filter = None
    if defect_class_filter:
        where_filter = {"defect_class": defect_class_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    passages: List[Dict[str, Any]] = []
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return passages

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance to cosine similarity score [0.0, 1.0]
        # In Chroma cosine distance: dist = 1 - cosine_similarity
        sim_score = max(0.0, min(1.0, 1.0 - float(dist)))
        passages.append({
            "id": meta.get("id", "UNKNOWN"),
            "defect_class": meta.get("defect_class", ""),
            "text": doc,
            "source": meta.get("source", "unspecified"),
            "score": round(sim_score, 4)
        })

    return passages


if __name__ == "__main__":
    test_query = "Fine micro-cracks network on hot rolled steel strip with roll thermal fatigue"
    print(f"Testing retrieval for: '{test_query}'")
    results = retrieve(test_query, k=2)
    for r in results:
        print(f"[{r['id']}] (score: {r['score']:.4f}) Source: {r['source']}")
        print(f"  {r['text']}\n")
