"""
Module: src/kb/build_index.py
Description: Builds and populates the vector index in ChromaDB using sentence-transformers (all-MiniLM-L6-v2).
             Reads passages from src/kb/passages.json and indexes them with full metadata.
Inputs: src/kb/passages.json
Outputs: Persistent ChromaDB collection at src/kb/chroma_db
# OWNER: Ravi (built by Aman for now)
# NOTE: Provisional metallurgical content — pending domain expert review by Tuhin.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).resolve().parent
PASSAGES_PATH = BASE_DIR / "passages.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "metallurgical_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_passages(filepath: Path = PASSAGES_PATH) -> List[Dict[str, Any]]:
    """Load metallurgical defect passages from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Passages file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        passages = json.load(f)
    return passages


def build_index(
    passages_path: Path = PASSAGES_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL_NAME
):
    """
    Build or rebuild the ChromaDB vector index from passages.json.
    Designed to be cleanly re-runnable whenever passages.json is updated.
    """
    print(f"Loading passages from {passages_path}...")
    passages = load_passages(passages_path)
    print(f"Found {len(passages)} passages.")

    print(f"Loading embedding model '{model_name}'...")
    embedder = SentenceTransformer(model_name)

    # Initialize persistent Chroma client
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Reset/recreate collection for clean re-runnability
    try:
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' already exists. Deleting for fresh rebuild...")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Note during collection check: {e}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [p["id"] for p in passages]
    texts = [p["text"] for p in passages]
    metadatas = [
        {
            "id": p["id"],
            "defect_class": p.get("defect_class", "unknown"),
            "source": p.get("source", "unspecified")
        }
        for p in passages
    ]

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    print(f"Indexing into ChromaDB collection '{collection_name}'...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    count = collection.count()
    print(f"Index successfully built! Total documents in collection: {count}")
    return count


if __name__ == "__main__":
    build_index()
