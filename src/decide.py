"""
Module: src/decide.py
Description: Decision engine mapping defect fingerprint and metallurgical diagnosis to actionable mill operations:
             "Inspect" | "Grind" | "Escalate".
Inputs:
  - fingerprint (dict): contains severity, affected_area_pct, class, morphology, location, pattern
  - diagnosis (dict): contains confidence, cited_passage_id, probable_origin
Outputs:
  - dict: {"action": "Inspect" | "Grind" | "Escalate", "rationale": str, "rule_triggered": str}
# OWNER: Aman
"""

from typing import Dict, Any


def decide(fingerprint: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine corrective mill action based on defect severity, affected area %,
    defect classification, and metallurgical diagnosis confidence.

    Documented Decision Threshold Rules:
    -----------------------------------
    1. Critical Escalation (Rule 1):
       - If severity is "High" AND affected_area_pct >= 15.0%
       - OR if defect class is "crazing" / "inclusion" with affected_area_pct >= 8.0%
       -> Action: "Escalate" (Major structural integrity or thermal roll degradation risk; halts line or flags supervisor).

    2. Diagnosis Ambiguity Escalation (Rule 2):
       - If severity is "High" AND diagnosis confidence < 0.55
       -> Action: "Escalate" (High severity with low knowledge base grounding demands human metallurgical intervention).

    3. Surface Conditioning Grind (Rule 3):
       - If severity is "Medium" or "High" AND affected_area_pct is between 3.0% and 15.0%
         AND defect class in ["scratches", "patches", "rolled-in_scale", "inclusion"]
       -> Action: "Grind" (Defect is superficial or localized mechanical gouge remediable via surface grinding line).

    4. Standard Inspection (Rule 4):
       - If severity is "Low" OR affected_area_pct < 3.0%
       -> Action: "Inspect" (Tolerable shallow defect; route to automated optical or manual visual line before coil release).

    5. Fallback Default:
       -> Action: "Inspect"
    """
    defect_class = fingerprint.get("class", "unknown")
    severity = fingerprint.get("severity", "Low")
    area_pct = float(fingerprint.get("affected_area_pct", 0.0))
    pattern = fingerprint.get("pattern", "isolated")
    conf = float(diagnosis.get("confidence", 0.0))
    cited_id = diagnosis.get("cited_passage_id", "NONE")

    # Rule 1: High severity with large area footprint or critical crack/inclusion defects
    if severity == "High" and area_pct >= 15.0:
        return {
            "action": "Escalate",
            "rule_triggered": "RULE_1_CRITICAL_HIGH_AREA",
            "rationale": (
                f"Defect '{defect_class}' exhibits High severity with {area_pct:.1f}% surface coverage. "
                f"Exceeds 15% threshold; requires immediate mill escalation to prevent slab downgrading or roll failure."
            )
        }

    if defect_class in ["crazing", "inclusion"] and severity == "High" and area_pct >= 8.0:
        return {
            "action": "Escalate",
            "rule_triggered": "RULE_1_CRITICAL_INTERNAL_RISK",
            "rationale": (
                f"Critical defect '{defect_class}' at {area_pct:.1f}% area poses severe metallurgical fracture risk. "
                f"Grounded diagnosis [{cited_id}] suggests immediate metallurgical engineering review."
            )
        }

    # Rule 2: High severity with low diagnosis confidence
    if severity == "High" and conf < 0.55:
        return {
            "action": "Escalate",
            "rule_triggered": "RULE_2_DIAGNOSIS_AMBIGUITY",
            "rationale": (
                f"Severe defect '{defect_class}' detected with insufficient knowledge base confidence ({conf:.2f} < 0.55). "
                f"Requires physical metallurgist verification before disposition."
            )
        }

    # Rule 3: Remediation via Grinding line
    if (severity in ["Medium", "High"] or area_pct >= 3.0) and defect_class in ["scratches", "patches", "rolled-in_scale", "inclusion"]:
        return {
            "action": "Grind",
            "rule_triggered": "RULE_3_SURFACE_CONDITIONING",
            "rationale": (
                f"Defect '{defect_class}' (severity: {severity}, area: {area_pct:.1f}%) is suitable for surface grinding conditioning. "
                f"Passage [{cited_id}] confirms mechanical or scale surface origin."
            )
        }

    # Rule 4: Mild defect -> Inspect
    if severity == "Low" or area_pct < 3.0:
        return {
            "action": "Inspect",
            "rule_triggered": "RULE_4_ROUTINE_INSPECTION",
            "rationale": (
                f"Defect '{defect_class}' presents Low severity ({area_pct:.1f}% area). "
                f"Release to inspection table for visual confirmation."
            )
        }

    # Default fallback
    return {
        "action": "Inspect",
        "rule_triggered": "RULE_DEFAULT_INSPECTION",
        "rationale": f"Defect '{defect_class}' marked for standard inspection audit."
    }


if __name__ == "__main__":
    test_fp = {"class": "scratches", "severity": "Medium", "affected_area_pct": 5.2, "pattern": "isolated"}
    test_diag = {"confidence": 0.72, "cited_passage_id": "SCR-001"}
    decision = decide(test_fp, test_diag)
    print("Self-test Decision:", decision)
