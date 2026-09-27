"""
Module: src/characterize.py
Description: Characterizes detected steel surface defects into structured metallurgical fingerprints.
             Computes morphology (elongated vs compact), location (edge vs center), affected area %,
             severity (Low/Medium/High), and patterns (isolated vs repetitive).
Inputs:
  - characterize: detection (dict: {"class", "bbox", "confidence"}), image_shape (tuple: (H, W) or (H, W, C))
  - detect_pattern: all_detections_in_image (list of detection dicts)
Outputs:
  - characterize: dict of fingerprint attributes
  - detect_pattern: str ("isolated" | "repetitive" | "none")
# OWNER: Aman
"""

from typing import Dict, Any, List, Tuple
from collections import Counter
from src.calibrate import calibrate_confidence


def characterize(detection: Dict[str, Any], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Compute morphological and spatial fingerprint for a single defect detection.

    Args:
        detection: Dict containing "class", "bbox" [x1, y1, x2, y2], and "confidence".
        image_shape: Image dimensions as (height, width) or (height, width, channels).

    Returns:
        Dict representing fingerprint:
            - class: defect class name
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - width: bbox width
            - height: bbox height
            - aspect_ratio: max(w, h) / min(w, h)
            - morphology: "elongated" if aspect_ratio > 2.5 else "compact"
            - centroid: (cx, cy)
            - location: "edge" if centroid within outer 15% margin else "center"
            - affected_area_pct: (bbox_area / image_area) * 100
            - severity: "High" | "Medium" | "Low"
    """
    img_h = float(image_shape[0])
    img_w = float(image_shape[1])
    img_area = max(img_h * img_w, 1.0)

    x1, y1, x2, y2 = [float(v) for v in detection["bbox"]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    bbox_area = w * h

    # 1. Morphology: aspect ratio calculation (invariant to horizontal/vertical orientation)
    min_dim = max(min(w, h), 1e-5)
    max_dim = max(w, h)
    aspect_ratio = max_dim / min_dim
    morphology = "elongated" if aspect_ratio > 2.5 else "compact"

    # 2. Location: centroid within outer 15% margin
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    norm_x = cx / img_w
    norm_y = cy / img_h

    # Outer 15% margin: 0.0 <= norm <= 0.15 or 0.85 <= norm <= 1.0
    if norm_x < 0.15 or norm_x > 0.85 or norm_y < 0.15 or norm_y > 0.85:
        location = "edge"
    else:
        location = "center"

    # 3. Affected Area Percentage
    affected_area_pct = (bbox_area / img_area) * 100.0

    # 4. Severity Rule:
    # Uses calibrated confidence from Temperature Scaling to prevent overconfident mis-scoring.
    # Weighted rule combining normalized affected area and calibrated confidence.
    # Area impact saturates at 15% coverage of total image area.
    # Score = 0.45 * calibrated_confidence + 0.55 * min(affected_area_pct / 15.0, 1.0)
    raw_conf = float(detection.get("confidence", 0.5))
    calibrated_conf = calibrate_confidence(raw_conf)
    area_factor = min(affected_area_pct / 15.0, 1.0)
    severity_score = (0.45 * calibrated_conf) + (0.55 * area_factor)

    if severity_score >= 0.60 or affected_area_pct >= 12.0:
        severity = "High"
    elif severity_score >= 0.35 or affected_area_pct >= 4.0:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "class": detection["class"],
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "confidence": round(raw_conf, 4),
        "calibrated_confidence": round(calibrated_conf, 4),
        "width": round(w, 2),
        "height": round(h, 2),
        "aspect_ratio": round(aspect_ratio, 2),
        "morphology": morphology,
        "centroid": (round(cx, 2), round(cy, 2)),
        "normalized_centroid": (round(norm_x, 3), round(norm_y, 3)),
        "location": location,
        "affected_area_pct": round(affected_area_pct, 2),
        "severity": severity,
        "severity_score": round(severity_score, 3)
    }


def detect_pattern(all_detections_in_image: List[Dict[str, Any]]) -> str:
    """
    Determine if detections represent a repetitive defect pattern across the surface
    or an isolated defect event.

    Args:
        all_detections_in_image: List of detection dicts (each having at least a "class" key).

    Returns:
        "repetitive" if 2+ detections share the same defect class,
        "isolated" if all classes appear at most once,
        "none" if no detections exist.
    """
    if not all_detections_in_image:
        return "none"

    class_counts = Counter(d["class"] for d in all_detections_in_image)
    if any(count >= 2 for count in class_counts.values()):
        return "repetitive"
    return "isolated"


if __name__ == "__main__":
    # Self-test with synthetic detection
    sample_det = {"class": "scratches", "bbox": [10.0, 20.0, 15.0, 180.0], "confidence": 0.88}
    shape = (200, 200, 3)
    fp = characterize(sample_det, shape)
    print("Sample Characterization:", fp)
    pattern = detect_pattern([sample_det, sample_det])
    print("Pattern check (2 of same):", pattern)
