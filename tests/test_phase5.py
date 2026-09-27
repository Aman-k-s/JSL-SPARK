"""
Test Phase 5: Decision Layer Evaluation.
Evaluates 6 varied fingerprint + diagnosis combinations covering Inspect, Grind, and Escalate.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.decide import decide

TEST_CASES = [
    {
        "name": "Case 1: Minor shallow scratch",
        "fp": {"class": "scratches", "severity": "Low", "affected_area_pct": 1.8, "pattern": "isolated"},
        "diag": {"confidence": 0.78, "cited_passage_id": "SCR-002", "probable_origin": "Superficial contact abrasion."}
    },
    {
        "name": "Case 2: Medium scratch line (dressable)",
        "fp": {"class": "scratches", "severity": "Medium", "affected_area_pct": 6.4, "pattern": "isolated"},
        "diag": {"confidence": 0.81, "cited_passage_id": "SCR-001", "probable_origin": "Side guide rubbing."}
    },
    {
        "name": "Case 3: Severe repetitive rolled-in scale",
        "fp": {"class": "rolled-in_scale", "severity": "High", "affected_area_pct": 18.2, "pattern": "repetitive"},
        "diag": {"confidence": 0.85, "cited_passage_id": "RIS-001", "probable_origin": "Descaling header failure."}
    },
    {
        "name": "Case 4: Critical crazing micro-cracks",
        "fp": {"class": "crazing", "severity": "High", "affected_area_pct": 11.5, "pattern": "isolated"},
        "diag": {"confidence": 0.79, "cited_passage_id": "CRZ-001", "probable_origin": "Work roll thermal fire cracking."}
    },
    {
        "name": "Case 5: Medium roll peel patch",
        "fp": {"class": "patches", "severity": "Medium", "affected_area_pct": 5.1, "pattern": "isolated"},
        "diag": {"confidence": 0.75, "cited_passage_id": "PAT-002", "probable_origin": "Work roll shelling peel."}
    },
    {
        "name": "Case 6: High severity with low diagnosis confidence (uncertain root cause)",
        "fp": {"class": "inclusion", "severity": "High", "affected_area_pct": 6.8, "pattern": "repetitive"},
        "diag": {"confidence": 0.42, "cited_passage_id": "INC-004", "probable_origin": "Uncertain exogenous origin."}
    }
]

def main():
    print("=" * 105)
    print("PHASE 5 VERIFICATION — DECISION LAYER ACTION & RATIONALE MATRIX")
    print("=" * 105)
    print(f"{'Case':<22} | {'Class':<15} | {'Severity':<8} | {'Area %':<7} | {'Diag Conf':<9} | {'ACTION':<9} | {'Rule Triggered'}")
    print("-" * 105)

    for tc in TEST_CASES:
        res = decide(tc["fp"], tc["diag"])
        print(f"{tc['name'][:21]:<22} | {tc['fp']['class']:<15} | {tc['fp']['severity']:<8} | {tc['fp']['affected_area_pct']:<7.1f} | {tc['diag']['confidence']:<9.2f} | {res['action']:<9} | {res['rule_triggered']}")
        print(f"  -> Rationale: {res['rationale']}\n")

    print("=" * 105)

if __name__ == "__main__":
    main()
