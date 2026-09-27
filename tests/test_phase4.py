"""
Test Phase 4: Metallurgical KB + Retrieval + LLM Grounded Diagnosis.
Runs 6 example fingerprints (one for each NEU-DET defect class) through diagnose()
and prints query -> retrieved passages -> generated diagnosis.
"""

import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnose import diagnose, build_query

SAMPLE_FINGERPRINTS = [
    {
        "class": "crazing",
        "morphology": "compact",
        "aspect_ratio": 2.02,
        "location": "edge",
        "affected_area_pct": 14.18,
        "severity": "High",
        "pattern": "isolated"
    },
    {
        "class": "inclusion",
        "morphology": "elongated",
        "aspect_ratio": 4.14,
        "location": "center",
        "affected_area_pct": 24.15,
        "severity": "High",
        "pattern": "repetitive"
    },
    {
        "class": "patches",
        "morphology": "compact",
        "aspect_ratio": 1.21,
        "location": "center",
        "affected_area_pct": 13.91,
        "severity": "High",
        "pattern": "repetitive"
    },
    {
        "class": "pitted_surface",
        "morphology": "compact",
        "aspect_ratio": 1.65,
        "location": "center",
        "affected_area_pct": 54.70,
        "severity": "High",
        "pattern": "repetitive"
    },
    {
        "class": "rolled-in_scale",
        "morphology": "compact",
        "aspect_ratio": 1.97,
        "location": "center",
        "affected_area_pct": 9.42,
        "severity": "Medium",
        "pattern": "repetitive"
    },
    {
        "class": "scratches",
        "morphology": "elongated",
        "aspect_ratio": 6.70,
        "location": "center",
        "affected_area_pct": 14.81,
        "severity": "High",
        "pattern": "repetitive"
    }
]

def main():
    print("=" * 85)
    print("PHASE 4 VERIFICATION — 6-CLASS GROUNDED METALLURGICAL DIAGNOSIS TEST")
    print("=" * 85)

    for i, fp in enumerate(SAMPLE_FINGERPRINTS, 1):
        print(f"\n[{i}/6] DEFECT CLASS: {fp['class'].upper()}")
        print("-" * 50)
        
        query = build_query(fp)
        print(f"Generated Query:\n  {query}\n")

        result = diagnose(fp, k=2)

        print("Retrieved Passages from ChromaDB:")
        for p in result["retrieved_passages"]:
            print(f"  * [{p['id']}] (Sim Score: {p['score']:.4f}) Source: {p['source']}")
            print(f"    Excerpt: {p['text'][:120]}...")

        print("\nLLM Grounded Diagnosis:")
        print(f"  Cited Passage ID: {result['cited_passage_id']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probable Origin:\n    {result['probable_origin']}")
        print("-" * 85)

if __name__ == "__main__":
    main()
