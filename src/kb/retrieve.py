"""
Module: src/kb/retrieve.py
Description: Vector similarity retrieval module over the metallurgical ChromaDB index.
             Returns top-k passages with evidence levels, source IDs, and relation types.
Inputs: query (str), optional k (int, default=3), optional defect_class_filter (str)
Outputs: list of dicts with {"id": str, "defect_class": str, "text": str, "source": str,
                            "source_ids": list, "evidence_level": str, "relation_type": str,
                            "diagnostic_use": str, "score": float}
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
        List of dicts: [
            {
                "id": str,
                "defect_class": str,
                "text": str,
                "source": str,
                "source_ids": list,
                "evidence_level": str,
                "relation_type": str,
                "diagnostic_use": str,
                "score": float
            }
        ]
    """
    collection, embedder = _get_resources()

    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

    where_filter = None
    if defect_class_filter:
        norm_class = defect_class_filter.replace("-", "_").lower()
        where_filter = {"defect_class": norm_class}

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        # Fallback to un-filtered retrieval if specific class query fails
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

    passages: List[Dict[str, Any]] = []
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return passages

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        sim_score = max(0.0, min(1.0, 1.0 - float(dist)))
        
        raw_source_ids = meta.get("source_ids", "")
        if isinstance(raw_source_ids, str):
            s_ids_list = [s.strip() for s in raw_source_ids.split(",") if s.strip()]
        else:
            s_ids_list = list(raw_source_ids)

        clean_text = meta.get("raw_text") or doc

        passages.append({
            "id": meta.get("id", "UNKNOWN"),
            "defect_class": meta.get("defect_class", ""),
            "text": clean_text,
            "source": meta.get("citations") or meta.get("source", "Technical Metallurgy Literature"),
            "source_ids": s_ids_list,
            "evidence_level": meta.get("evidence_level", "supported_inference"),
            "relation_type": meta.get("relation_type", ""),
            "diagnostic_use": meta.get("diagnostic_use", ""),
            "score": round(sim_score, 4)
        })

    return passages


if __name__ == "__main__":
    test_query = "Linear gouge on steel surface from mechanical slide contact with side guide"
    print(f"Testing retrieval for: '{test_query}'")
    results = retrieve(test_query, k=2)
    for r in results:
        print(f"[{r['id']}] (Sim: {r['score']:.4f} | Level: {r['evidence_level']}) Sources: {r['source_ids']}")
        print(f"  {r['text']}")
        print(f"  Ref: {r['source']}\n")
