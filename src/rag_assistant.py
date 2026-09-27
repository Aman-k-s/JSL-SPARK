"""
Module: src/rag_assistant.py
Description: Interactive Metallurgical RAG (Retrieval-Augmented Generation) Q&A Assistant.
             Enables users/metallurgists to query the steel defect knowledge base interactively,
             optionally contextualized with the currently inspected defect and process telemetry.
Inputs: user_query (str), optional current_fingerprint (dict), optional current_diagnosis (dict), optional k (int)
Outputs: dict with {"answer": str, "cited_passages": list, "query": str}
# OWNER: Aman (temporary, will hand off to Tuhin & Ravi)
"""

import os
import json
import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.kb.retrieve import retrieve


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


def _call_groq_chat(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Groq API with robust model fallback."""
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
                    temperature=0.2,
                    max_tokens=600
                )
                if response and response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
            except Exception:
                continue
    except Exception as e:
        print(f"Groq API error in RAG assistant: {e}")
    return None


def answer_question(
    user_query: str,
    current_fingerprint: Optional[Dict[str, Any]] = None,
    current_diagnosis: Optional[Dict[str, Any]] = None,
    k: int = 4
) -> Dict[str, Any]:
    """
    Answer a metallurgical question using vector retrieval from the knowledge base.

    Args:
        user_query: Natural language query from the metallurgist or mill operator.
        current_fingerprint: Optional active defect fingerprint from current inspection.
        current_diagnosis: Optional active diagnosis from current inspection.
        k: Number of passages to retrieve.

    Returns:
        Dict:
            - answer: Synthesized grounded response
            - cited_passages: List of retrieved passages used
            - query: Original query
    """
    # Augment retrieval search if user references the current image defect
    search_query = user_query
    if current_fingerprint and any(w in user_query.lower() for w in ["this", "current", "here", "detected", "defect"]):
        search_query = f"{user_query} (Context: defect class '{current_fingerprint.get('class')}', severity '{current_fingerprint.get('severity')}')"

    # Retrieve candidate passages
    retrieved = retrieve(search_query, k=k)

    if not retrieved:
        return {
            "answer": "No relevant metallurgical knowledge base passages could be retrieved for this query.",
            "cited_passages": [],
            "query": user_query
        }

    # Format passages for context including evidence levels and source IDs
    context_blocks = []
    for p in retrieved:
        context_blocks.append(
            f"[{p['id']}] (Defect Class: {p['defect_class']} | Similarity: {p['score']:.3f})\n"
            f"Evidence Level: {p.get('evidence_level', 'supported_inference')} | Source IDs: {p.get('source_ids', [])}\n"
            f"Technical References: {p.get('source', '')}\n"
            f"Mechanism Content: {p['text']}\n"
            f"Diagnostic & Investigation Guidance: {p.get('diagnostic_use', '')}"
        )
    context_str = "\n\n".join(context_blocks)

    # Format optional inspection context
    inspection_ctx = ""
    if current_fingerprint:
        inspection_ctx = (
            f"\nCURRENT SPECIMEN INSPECTION CONTEXT:\n"
            f"- Classified Defect: {current_fingerprint.get('class')}\n"
            f"- Confidence: {current_fingerprint.get('confidence')}\n"
            f"- Morphology: {current_fingerprint.get('morphology')} (aspect ratio: {current_fingerprint.get('aspect_ratio')})\n"
            f"- Strip Location: {current_fingerprint.get('location')}\n"
            f"- Affected Area: {current_fingerprint.get('affected_area_pct')}%\n"
            f"- Severity Level: {current_fingerprint.get('severity')}\n"
        )
        if current_diagnosis:
            inspection_ctx += (
                f"- Grounded Probable Origin: {current_diagnosis.get('probable_origin_hypothesis') or current_diagnosis.get('probable_origin')}\n"
                f"- Recommended Investigation: {current_diagnosis.get('recommended_investigation')}\n"
                f"- Cited Passage: [{current_diagnosis.get('cited_passage_id')}] Sources: {current_diagnosis.get('source_ids')}\n"
            )

    system_prompt = (
        "You are an expert metallurgical quality assistant for hot-rolled steel manufacturing.\n"
        "STRICT GUIDELINES & DIAGNOSTIC POLICY:\n"
        "1. Answer the user's question clearly, professionally, and helpfully.\n"
        "2. Base your answer STRICTLY on the facts and mechanisms in the provided metallurgical passages.\n"
        "3. Frame causes as PROBABLE-ORIGIN HYPOTHESES (using 'consistent with', 'probable origin', 'requires validation') rather than claiming absolute root cause from visual appearance alone.\n"
        "4. Include concrete RECOMMENDED INVESTIGATION actions (e.g. roll wear audits, descaling header inspections, chemistry logs).\n"
        "5. Explicitly cite passage IDs (e.g. [SCR-01], [RIS-05]) and technical source IDs (e.g. [S2, S7]) in your response.\n"
        "6. If the provided passages do not contain enough information to address a specific aspect of the question, state that clearly rather than inventing facts.\n"
        "7. If current inspection context is provided and relevant, reference it directly."
    )

    user_prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"{inspection_ctx}\n"
        f"RETRIEVED KNOWLEDGE BASE PASSAGES:\n{context_str}\n\n"
        f"Provide a grounded, expert metallurgical response citing specific passage IDs:"
    )

    raw_answer = _call_groq_chat(system_prompt, user_prompt)

    if not raw_answer:
        # Fallback summary of top passage if API is unavailable
        top_p = retrieved[0]
        raw_answer = (
            f"Based on passage [{top_p['id']}] ({top_p['defect_class']}):\n\n"
            f"{top_p['text']}\n\n"
            f"*(Source: {top_p['source']})*"
        )

    return {
        "answer": raw_answer,
        "cited_passages": retrieved,
        "query": user_query
    }


if __name__ == "__main__":
    test_q = "What causes rolled-in scale and what descaling pressure is required?"
    print(f"Testing answer_question for: '{test_q}'")
    resp = answer_question(test_q)
    print("\nASSISTANT ANSWER:\n", resp["answer"])
    print("\nCITATIONS:")
    for p in resp["cited_passages"]:
        print(f" - [{p['id']}] (score: {p['score']:.3f})")
