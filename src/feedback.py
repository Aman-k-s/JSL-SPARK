"""
Module: src/feedback.py
Description: Active feedback loop enabling metallurgists and operators to confirm or correct diagnoses.
             On correction, dynamically appends a new passage to passages.json tagged as
             source="user_correction", status="pending_review" and triggers ChromaDB index rebuild.
Inputs:
  - record_confirmation: fingerprint (dict), diagnosis (dict)
  - record_correction: defect_class (str), corrected_origin (str), fingerprint (dict), original_diagnosis (dict)
Outputs: dict with success status, feedback ID, and updated passage count
# OWNER: Aman
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kb.build_index import build_index

PASSAGES_PATH = PROJECT_ROOT / "src" / "kb" / "passages.json"
FEEDBACK_LOG_PATH = PROJECT_ROOT / "src" / "feedback_log.json"


def _load_json(filepath: Path, default: Any):
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(filepath: Path, data: Any):
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_confirmation(fingerprint: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log an operator/metallurgist confirmation of a correct diagnosis.
    """
    log = _load_json(FEEDBACK_LOG_PATH, [])
    entry = {
        "id": f"CONF-{len(log) + 1:04d}",
        "timestamp": datetime.now().isoformat(),
        "type": "confirmation",
        "defect_class": fingerprint.get("class"),
        "cited_passage_id": diagnosis.get("cited_passage_id"),
        "confidence": diagnosis.get("confidence"),
        "fingerprint": fingerprint,
        "probable_origin": diagnosis.get("probable_origin")
    }
    log.append(entry)
    _save_json(FEEDBACK_LOG_PATH, log)
    return {"status": "success", "event_id": entry["id"], "type": "confirmation"}


def record_correction(
    defect_class: str,
    corrected_origin: str,
    fingerprint: Dict[str, Any],
    original_diagnosis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Record an expert domain correction, append to passages.json as a provisional passage,
    and trigger an automatic re-indexing of ChromaDB.

    Args:
        defect_class: Corrected or confirmed defect class.
        corrected_origin: Domain expert explanation of the true root cause.
        fingerprint: The defect characterization metrics.
        original_diagnosis: The previous system diagnosis that was corrected.

    Returns:
        Dict: {"status": "success", "passage_id": str, "total_passages": int}
    """
    passages = _load_json(PASSAGES_PATH, [])
    
    # Calculate next correction ID (e.g. CORR-001)
    existing_corr_ids = [p["id"] for p in passages if p.get("id", "").startswith("CORR-")]
    next_num = len(existing_corr_ids) + 1
    new_passage_id = f"CORR-{next_num:03d}"

    new_passage = {
        "id": new_passage_id,
        "defect_class": defect_class,
        "text": corrected_origin.strip(),
        "source": "user_correction",
        "status": "pending_review",
        "submitted_at": datetime.now().isoformat()
    }

    passages.append(new_passage)
    _save_json(PASSAGES_PATH, passages)

    # Re-run build_index to update ChromaDB vector store
    print(f"Re-indexing ChromaDB with new corrected passage [{new_passage_id}]...")
    total_docs = build_index()

    # Also log to feedback log
    log = _load_json(FEEDBACK_LOG_PATH, [])
    log_entry = {
        "id": f"FEEDBACK-{len(log) + 1:04d}",
        "timestamp": datetime.now().isoformat(),
        "type": "correction",
        "passage_id": new_passage_id,
        "defect_class": defect_class,
        "corrected_origin": corrected_origin,
        "original_diagnosis": original_diagnosis,
        "fingerprint": fingerprint
    }
    log.append(log_entry)
    _save_json(FEEDBACK_LOG_PATH, log)

    return {
        "status": "success",
        "passage_id": new_passage_id,
        "total_passages": total_docs,
        "message": f"Successfully registered correction [{new_passage_id}] and re-indexed knowledge base."
    }


if __name__ == "__main__":
    print("Testing active feedback loop...")
    test_fp = {"class": "scratches", "severity": "Medium"}
    test_diag = {"cited_passage_id": "SCR-001", "confidence": 0.70}
    res = record_confirmation(test_fp, test_diag)
    print("Confirmation logged:", res)
