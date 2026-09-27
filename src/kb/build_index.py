"""
Module: src/kb/build_index.py
Description: Builds and populates the vector index in ChromaDB using sentence-transformers (all-MiniLM-L6-v2).
             Loads the 48 curated metallurgical passages authored by Tuhin from steel_surface_metallurgy_rag_kb_final.json,
             complete with 11 academic/technical source mappings and evidence levels.
Inputs: src/kb/steel_surface_metallurgy_rag_kb_final.json
Outputs: Persistent ChromaDB collection at src/kb/chroma_db
# OWNER: Ravi & Tuhin (built by Aman)
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
KB_JSON_PATH = BASE_DIR / "steel_surface_metallurgy_rag_kb_final.json"
FALLBACK_KB_JSON_PATH = BASE_DIR.parent.parent / "steel_surface_metallurgy_rag_kb_final.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "metallurgical_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PASSAGES_COMPAT_PATH = BASE_DIR / "passages.json"


def load_tuhin_kb(filepath: Path = KB_JSON_PATH) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Load curated metallurgical knowledge base from Tuhin's final JSON file.
    Returns (passages, sources, diagnostic_policy).
    """
    target = filepath if filepath.exists() else FALLBACK_KB_JSON_PATH
    if not target.exists():
        raise FileNotFoundError(f"Knowledge base file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    passages = kb_data.get("passages", [])
    sources = kb_data.get("sources", {})
    diagnostic_policy = kb_data.get("diagnostic_policy", {})
    return passages, sources, diagnostic_policy


def format_source_citation(source_ids: List[str], sources: Dict[str, Any]) -> str:
    """Format short human-readable citations from source IDs."""
    citations = []
    for sid in source_ids:
        s_info = sources.get(sid, {})
        title = s_info.get("title", sid)
        year = s_info.get("year", "")
        authors = s_info.get("authors", "").split(";")[0]
        if authors and year:
            citations.append(f"{sid}: {authors} ({year})")
        elif year:
            citations.append(f"{sid}: {title[:30]} ({year})")
        else:
            citations.append(f"{sid}: {title[:35]}")
    return "; ".join(citations) if citations else "Internal Metallurgy Knowledge"


def build_index(
    kb_path: Path = KB_JSON_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL_NAME
):
    """
    Build or rebuild the ChromaDB vector index from Tuhin's curated metallurgical KB.
    Indexes all 48 passages with rich relation types, evidence levels, and source citations.
    """
    print(f"Loading curated knowledge base from {kb_path}...")
    passages, sources, policy = load_tuhin_kb(kb_path)
    print(f"Found {len(passages)} passages backed by {len(sources)} technical sources.")

    # Save a clean passages copy for backward compatibility
    compat_records = []
    for p in passages:
        compat_records.append({
            "id": p["id"],
            "defect_class": p.get("defect_class", ""),
            "text": p["text"],
            "source": format_source_citation(p.get("source_ids", []), sources),
            "relation_type": p.get("relation_type", ""),
            "evidence_level": p.get("evidence_level", ""),
            "diagnostic_use": p.get("diagnostic_use", ""),
            "source_ids": p.get("source_ids", [])
        })
    with open(PASSAGES_COMPAT_PATH, "w", encoding="utf-8") as f:
        json.dump(compat_records, f, indent=2, ensure_ascii=False)

    print(f"Loading embedding model '{model_name}'...")
    embedder = SentenceTransformer(model_name)

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Reset collection for clean re-runnability
    try:
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' exists. Deleting for fresh rebuild...")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Note during collection reset: {e}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    metadatas = []

    for p in passages:
        pid = p["id"]
        defect_cls = p.get("defect_class", "").replace("-", "_")
        rel_type = p.get("relation_type", "")
        keywords = ", ".join(p.get("trigger_keywords", []))
        text = p["text"]
        src_ids = p.get("source_ids", [])
        citations_str = format_source_citation(src_ids, sources)

        # Semantic document content optimized for retrieval queries
        doc_text = (
            f"Defect Class: {defect_cls} | Relation: {rel_type} | Keywords: {keywords}\n"
            f"{text}\n"
            f"Diagnostic Guidance: {p.get('diagnostic_use', '')}"
        )

        ids.append(pid)
        documents.append(doc_text)
        metadatas.append({
            "id": pid,
            "defect_class": defect_cls,
            "raw_text": text,
            "relation_type": rel_type,
            "evidence_level": p.get("evidence_level", "supported_inference"),
            "source_ids": ", ".join(src_ids),
            "citations": citations_str,
            "diagnostic_use": p.get("diagnostic_use", "")
        })

    print(f"Computing embeddings for {len(documents)} passages...")
    embeddings = embedder.encode(documents, convert_to_numpy=True).tolist()

    print(f"Indexing into ChromaDB collection '{collection_name}'...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    count = collection.count()
    print(f"Index successfully built! Total documents indexed: {count}")
    return count


if __name__ == "__main__":
    build_index()
