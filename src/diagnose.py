"""
Module: src/diagnose.py
Description: Generates metallurgical probable-origin hypothesis and recommended investigation actions
             strictly grounded in 48 curated domain knowledge base passages.
             Adheres to metallurgical diagnostic policy: avoids declaring definitive root causes from images alone.
Inputs: fingerprint (dict from characterize.py)
Outputs: dict with probable_origin, probable_origin_hypothesis, recommended_investigation,
         cited_passage_id, source_ids, evidence_level, confidence, retrieved_passages.
# OWNER: Metallurgical Quality Systems Engineering
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env file from workspace root if present
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from src.kb.retrieve import retrieve


def build_query(fingerprint: Dict[str, Any]) -> str:
    """
    Template-fill fingerprint fields into a descriptive natural language query
    tailored for metallurgical knowledge base vector retrieval.
    """
    defect_class = fingerprint.get("class", "unknown defect")
    morphology = fingerprint.get("morphology", "compact")
    location = fingerprint.get("location", "center")
    area_pct = fingerprint.get("affected_area_pct", 0.0)
    severity = fingerprint.get("severity", "Medium")
    pattern = fingerprint.get("pattern", "isolated")
    aspect_ratio = fingerprint.get("aspect_ratio", 1.0)

    query = (
        f"Hot-rolled steel surface defect identified as '{defect_class}'. "
        f"Morphology is {morphology} with aspect ratio {aspect_ratio:.1f}, "
        f"spatial location is at the {location} of the strip, "
        f"covering {area_pct:.1f}% affected surface area with {severity} severity, "
        f"occurring in an {pattern} pattern. "
        f"Defect mechanism, roll wear, guide interaction, scale formation, or non-metallic inclusion origin."
    )
    return query


def _get_groq_api_key() -> Optional[str]:
    """Retrieve Groq API key from environment variable or Streamlit Cloud secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return key


def _call_groq_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Groq API using groq Python package with fallback models."""
    api_key = _get_groq_api_key()
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        models_to_try = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ]
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                if response and response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
            except Exception:
                continue
    except Exception as e:
        print(f"Groq API call encountered: {e}")
    return None


def diagnose(
    fingerprint: Dict[str, Any],
    k: int = 3
) -> Dict[str, Any]:
    """
    Diagnose metallurgical origin hypothesis and recommended investigation action using curated domain KB.

    Policy Constraints (Metallurgical Diagnostic Policy):
    - Output a probable-origin hypothesis, not a definitive root-cause claim, from visual evidence alone.
    - Always cite the retrieved passage ID and technical source IDs.
    - Recommend concrete plant investigation actions to validate the mechanism.
    - Confidence derived directly from vector similarity score (0.0 - 1.0).
    """
    query = build_query(fingerprint)
    defect_class = fingerprint.get("class", "").replace("-", "_")
    
    # Retrieve top passages from ChromaDB
    retrieved = retrieve(query, k=k, defect_class_filter=defect_class)
    if not retrieved:
        retrieved = retrieve(query, k=k)

    if not retrieved:
        return {
            "probable_origin": "No matching metallurgical knowledge base passages found to support a diagnosis.",
            "probable_origin_hypothesis": "No matching metallurgical knowledge base passages found to support a diagnosis.",
            "recommended_investigation": "Perform standard visual inspection and review recent coil genealogy.",
            "cited_passage_id": "NONE",
            "source_ids": [],
            "evidence_level": "none",
            "source_citations": "Unspecified",
            "confidence": 0.0,
            "retrieved_passages": [],
            "query_used": query
        }

    top_score = retrieved[0]["score"]

    # Build context for LLM with curated passages and sources
    context_str = "\n\n".join([
        f"[Passage ID: {p['id']}]\n"
        f"Relation Type: {p.get('relation_type', '')} | Evidence Level: {p.get('evidence_level', '')}\n"
        f"Source IDs: {p.get('source_ids', [])} | References: {p.get('source', '')}\n"
        f"Passage Content: {p['text']}\n"
        f"Diagnostic Guidance: {p.get('diagnostic_use', '')}"
        for p in retrieved
    ])

    system_prompt = (
        "You are an expert metallurgical quality diagnostic system for hot-rolled steel manufacturing.\n"
        "STRICT CONSTRAINTS & DIAGNOSTIC POLICY:\n"
        "1. Answer ONLY using the facts provided in the passages below.\n"
        "2. Do NOT declare a definitive root cause from visual appearance alone. Output a PROBABLE-ORIGIN HYPOTHESIS using cautious language ('consistent with', 'probable origin', 'requires validation').\n"
        "3. Provide a concrete RECOMMENDED INVESTIGATION action (e.g. check descaling pressure, inspect roll barrel, check side guides, slab scarfing logs).\n"
        "4. You MUST cite which passage ID you used as the primary basis (e.g. 'SCR-01', 'RIS-07', 'CRZ-02').\n"
        "5. Return your answer strictly as a JSON object with keys:\n"
        "   \"probable_origin_hypothesis\": (1-3 sentences stating the probable mechanism consistent with visual features),\n"
        "   \"recommended_investigation\": (concrete plant/lab investigation action to validate root cause),\n"
        "   \"cited_passage_id\": (the exact passage ID string cited),\n"
        "   \"alternative_hypotheses\": (secondary plausible mechanism if applicable, or 'None')\n"
    )

    user_prompt = (
        f"Specimen Fingerprint Metrics:\n{json.dumps(fingerprint, indent=2)}\n\n"
        f"Retrieved Metallurgical Passages from Knowledge Base:\n{context_str}\n\n"
        f"Formulate the probable-origin hypothesis and recommended investigation action strictly from the passages above."
    )

    llm_output_raw = _call_groq_llm(system_prompt, user_prompt)
    
    hypothesis = None
    investigation = None
    cited_id = None
    alt_hyp = None

    if llm_output_raw:
        try:
            data = json.loads(llm_output_raw)
            hypothesis = data.get("probable_origin_hypothesis") or data.get("probable_origin")
            investigation = data.get("recommended_investigation")
            cited_id = data.get("cited_passage_id")
            alt_hyp = data.get("alternative_hypotheses")
        except Exception:
            id_match = re.search(r'"cited_passage_id":\s*"([^"]+)"', llm_output_raw)
            if id_match:
                cited_id = id_match.group(1)
            hyp_match = re.search(r'"probable_origin_hypothesis":\s*"([^"]+)"', llm_output_raw)
            if hyp_match:
                hypothesis = hyp_match.group(1)
            inv_match = re.search(r'"recommended_investigation":\s*"([^"]+)"', llm_output_raw)
            if inv_match:
                investigation = inv_match.group(1)

    # Fallback to direct top retrieved passage if LLM API is unavailable
    if not hypothesis or not cited_id:
        top_passage = retrieved[0]
        cited_id = top_passage["id"]
        hypothesis = f"Visual features are consistent with mechanism described in passage [{cited_id}]: {top_passage['text']}"
        investigation = f"Validate via operational audit: {top_passage.get('diagnostic_use', 'Inspect roll condition and process parameters.')}"

    # Ensure cited_id exists in retrieved passages, else default to top passage
    valid_ids = [p["id"] for p in retrieved]
    if cited_id not in valid_ids:
        cited_id = retrieved[0]["id"]

    matched_p = next((p for p in retrieved if p["id"] == cited_id), retrieved[0])
    source_ids = matched_p.get("source_ids", [])
    evidence_level = matched_p.get("evidence_level", "supported_inference")
    citations = matched_p.get("source", "Technical Metallurgy Literature")

    return {
        "probable_origin": hypothesis,
        "probable_origin_hypothesis": hypothesis,
        "recommended_investigation": investigation or "Review roll wear logs and descaling records.",
        "alternative_hypotheses": alt_hyp or "None documented in active passages.",
        "cited_passage_id": cited_id,
        "source_ids": source_ids,
        "evidence_level": evidence_level,
        "source_citations": citations,
        "confidence": top_score,
        "retrieved_passages": retrieved,
        "query_used": query
    }


if __name__ == "__main__":
    sample_fp = {
        "class": "rolled-in_scale",
        "morphology": "compact",
        "aspect_ratio": 1.9,
        "location": "center",
        "affected_area_pct": 9.4,
        "severity": "Medium",
        "pattern": "repetitive"
    }
    print("Testing diagnose() with curated 48-passage domain KB...")
    res = diagnose(sample_fp)
    print("\nPROBABLE-ORIGIN HYPOTHESIS:\n", res["probable_origin_hypothesis"])
    print("\nRECOMMENDED INVESTIGATION:\n", res["recommended_investigation"])
    print(f"\nCITED PASSAGE: [{res['cited_passage_id']}] (Confidence: {res['confidence']:.4f})")
    print(f"SOURCE IDS: {res['source_ids']} | LEVEL: {res['evidence_level']}")
    print(f"CITATIONS: {res['source_citations']}")
