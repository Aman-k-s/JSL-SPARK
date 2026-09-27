"""
Module: src/process_correlate.py
Description: Correlates hot strip mill process telemetry (roll speed, reheat temp, descaling pressure,
             chemistry) with defect fingerprints to illuminate plant-level root causes.
             NOTE: Simulated example — not real production data.
Inputs: defect_class (str), optional batch_id (str)
Outputs: dict with process parameters, metallurgical correlation mechanism, and risk alerts
# OWNER: Mill Process Metallurgy Node
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "src" / "kb" / "synthetic_process_data.json"

_CACHED_BATCHES = None


def load_process_batches() -> List[Dict[str, Any]]:
    """Load synthetic process telemetry batches from JSON."""
    global _CACHED_BATCHES
    if _CACHED_BATCHES is None:
        if not SYNTHETIC_DATA_PATH.exists():
            return []
        with open(SYNTHETIC_DATA_PATH, "r", encoding="utf-8") as f:
            _CACHED_BATCHES = json.load(f)
    return _CACHED_BATCHES


def get_batch_by_id(batch_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve process data for a specific coil batch ID."""
    batches = load_process_batches()
    for b in batches:
        if b["batch_id"].lower() == batch_id.lower():
            return b
    return None


def correlate_process_parameters(defect_class: str, batch_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Correlate defect classification with hot rolling process telemetry.

    Args:
        defect_class: The classified defect name (e.g. 'scratches', 'rolled-in_scale').
        batch_id: Optional specific batch ID. If None, retrieves matching prototypical batch.

    Returns:
        Dict:
            - is_simulated: True (labeled clearly as simulated)
            - label: "Simulated example — not real production data"
            - batch_id: str
            - steel_grade: str
            - key_telemetry: dict of process values
            - correlation_mechanism: explanation grounded in metallurgical kinetics
            - anomalous_parameters: list of flagged out-of-spec parameters
    """
    batches = load_process_batches()
    matched_batch = None

    if batch_id:
        matched_batch = get_batch_by_id(batch_id)

    if not matched_batch:
        # Find prototypical batch correlating to this defect class
        for b in batches:
            if b.get("correlated_defect_risk") == defect_class:
                matched_batch = b
                break

    if not matched_batch:
        # Generic fallback
        matched_batch = batches[0] if batches else {}

    # Identify out-of-spec parameters
    anomalies = []
    furnace_temp = matched_batch.get("furnace_exit_temp_c", 1200)
    descaling_pres = matched_batch.get("descaling_pressure_bar", 200)
    roll_speed = matched_batch.get("roll_speed_mpm", 800)
    tonnage = matched_batch.get("roll_campaign_tonnage", 500)
    chem = matched_batch.get("chemistry_wt_pct", {})

    if furnace_temp > 1250:
        anomalies.append(f"Excessive furnace exit temp ({furnace_temp}°C > 1250°C)")
    if descaling_pres < 160:
        anomalies.append(f"Low descaling header pressure ({descaling_pres} bar < 180 bar nominal)")
    if roll_speed > 1000:
        anomalies.append(f"High strip transit speed ({roll_speed} mpm > 1000 mpm)")
    if tonnage > 1600:
        anomalies.append(f"Extended roll campaign ({tonnage} tons > 1500 t schedule)")
    if chem.get("Cu", 0.0) > 0.20:
        anomalies.append(f"Elevated tramp copper ({chem['Cu']} wt% Cu > 0.20 wt%)")
    if chem.get("total_oxygen_ppm", 0) > 25:
        anomalies.append(f"High total oxygen ({chem['total_oxygen_ppm']} ppm > 25 ppm)")

    return {
        "is_simulated": True,
        "disclaimer": "Simulated example — not real production data",
        "batch_id": matched_batch.get("batch_id", "UNKNOWN"),
        "steel_grade": matched_batch.get("steel_grade", "Standard Steel"),
        "telemetry": {
            "Furnace Exit Temp (°C)": furnace_temp,
            "Finishing Entry Temp (°C)": matched_batch.get("finishing_entry_temp_c", 980),
            "Roll Speed (mpm)": roll_speed,
            "Descaling Pressure (bar)": descaling_pres,
            "Roll Campaign Tonnage (t)": tonnage,
            "Chemistry (wt%)": chem
        },
        "correlation_mechanism": matched_batch.get(
            "correlation_mechanism",
            "Nominal process parameters detected; defect likely driven by localized mechanical or surface event."
        ),
        "anomalous_parameters": anomalies
    }


if __name__ == "__main__":
    print("Testing process correlation module...")
    corr = correlate_process_parameters("rolled-in_scale")
    print(f"Batch: {corr['batch_id']} | Grade: {corr['steel_grade']}")
    print(f"Disclaimer: {corr['disclaimer']}")
    print(f"Mechanism: {corr['correlation_mechanism']}")
    print(f"Flagged Anomalies: {corr['anomalous_parameters']}")
