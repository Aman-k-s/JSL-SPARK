"""
Module: src/diagnose.py
Description: Generates metallurgical root-cause diagnosis grounded strictly in retrieved knowledge base passages.
             Uses Groq LLM (e.g. llama-3.3-70b-versatile / llama-3.1-8b-instant) with strict anti-hallucination prompting.
Inputs: fingerprint (dict from characterize.py)
Outputs: dict with {"probable_origin": str, "cited_passage_id": str, "confidence": float, "retrieved_passages": list}
# OWNER: Aman (temporary, will hand off to Tuhin)
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load .env file from workspace root if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
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
        f"Metallurgical root causes, roll condition, scale, inclusions, or mechanical mill origin."
    )
    return query


def _call_groq_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Groq API using groq Python package with fallback models."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Models available on this account: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b
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
                    max_tokens=400,
                    response_format={"type": "json_object"}
                )
                if response and response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
            except Exception as e:
                # Try next model in sequence
                continue
    except Exception as e:
        print(f"Groq API call encountered: {e}")
    return None


def diagnose(
    fingerprint: Dict[str, Any],
    k: int = 3
) -> Dict[str, Any]:
    """
    Diagnose metallurgical origin using retrieved passages and strict grounded reasoning.

    Args:
        fingerprint: Dict with defect metrics from characterize.py.
        k: Number of candidate passages to retrieve.

    Returns:
        Dict:
            - probable_origin: Grounded metallurgical diagnosis statement
            - cited_passage_id: Exact passage ID cited (e.g. 'SCR-001')
            - confidence: Confidence derived directly from vector similarity score (0.0 - 1.0)
            - retrieved_passages: Top-k passages with scores
            - query_used: Natural language query passed to retrieval
    """
    query = build_query(fingerprint)
    defect_class = fingerprint.get("class")
    
    # Retrieve top passages from ChromaDB
    retrieved = retrieve(query, k=k, defect_class_filter=defect_class)
    if not retrieved:
        # If class-filtered retrieval yields nothing, retrieve globally
        retrieved = retrieve(query, k=k)

    if not retrieved:
        return {
            "probable_origin": "No matching metallurgical knowledge base passages found to support a diagnosis.",
            "cited_passage_id": "NONE",
            "confidence": 0.0,
            "retrieved_passages": [],
            "query_used": query
        }

    # Best similarity score from retrieval provides the confidence score (per requirements)
    top_score = retrieved[0]["score"]

    # Prepare context for LLM
    context_str = "\n\n".join([
        f"[Passage ID: {p['id']}]\nClass: {p['defect_class']}\nSource: {p['source']}\nContent: {p['text']}"
        for p in retrieved
    ])

    system_prompt = (
        "You are an expert metallurgical failure diagnosis system for hot-rolled steel.\n"
        "STRICT CONSTRAINTS:\n"
        "1. Answer ONLY using the facts provided in the passages below.\n"
        "2. Do NOT invent, assume, or extrapolate any mechanisms or statistics.\n"
        "3. You MUST cite which passage ID you used as the primary basis.\n"
        "4. If the provided passages do NOT support a confident answer, say so explicitly instead of guessing.\n"
        "5. Return your answer ONLY as a JSON object with keys:\n"
        "   \"probable_origin\": (concise 1-3 sentence metallurgical root-cause explanation),\n"
        "   \"cited_passage_id\": (the exact passage ID string cited, e.g. 'SCR-001')\n"
    )

    user_prompt = (
        f"Defect Fingerprint:\n{json.dumps(fingerprint, indent=2)}\n\n"
        f"Available Metallurgical Passages:\n{context_str}\n\n"
        f"Determine the probable metallurgical origin and cite the exact passage ID."
    )

    llm_output_raw = _call_groq_llm(system_prompt, user_prompt)
    
    probable_origin = None
    cited_passage_id = None

    if llm_output_raw:
        try:
            data = json.loads(llm_output_raw)
            probable_origin = data.get("probable_origin")
            cited_passage_id = data.get("cited_passage_id")
        except Exception:
            # Fallback regex extraction if raw json formatting was slightly off
            id_match = re.search(r'"cited_passage_id":\s*"([^"]+)"', llm_output_raw)
            origin_match = re.search(r'"probable_origin":\s*"([^"]+)"', llm_output_raw)
            if id_match:
                cited_passage_id = id_match.group(1)
            if origin_match:
                probable_origin = origin_match.group(1)

    # Fallback to direct top retrieved passage if LLM API was unreachable
    if not probable_origin or not cited_passage_id:
        top_passage = retrieved[0]
        cited_passage_id = top_passage["id"]
        probable_origin = (
            f"Based on grounded passage {top_passage['id']}, this defect is attributed to: "
            f"{top_passage['text']}"
        )

    # Validate that cited_passage_id matches one of the retrieved passages
    valid_ids = [p["id"] for p in retrieved]
    if cited_passage_id not in valid_ids:
        cited_passage_id = retrieved[0]["id"]

    return {
        "probable_origin": probable_origin,
        "cited_passage_id": cited_passage_id,
        "confidence": top_score,
        "retrieved_passages": retrieved,
        "query_used": query
    }


if __name__ == "__main__":
    sample_fp = {
        "class": "scratches",
        "morphology": "elongated",
        "aspect_ratio": 6.7,
        "location": "center",
        "affected_area_pct": 14.8,
        "severity": "High",
        "pattern": "repetitive"
    }
    print("Testing diagnose() on sample scratch fingerprint...")
    res = diagnose(sample_fp)
    print("\nDIAGNOSIS RESULT:")
    print("Cited Passage ID:", res["cited_passage_id"])
    print("Confidence (from retrieval):", res["confidence"])
    print("Probable Origin:", res["probable_origin"])
