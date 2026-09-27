"""
End-to-End automated integration test verifying full pipeline execution across 3+ real test images.
Tests: detect -> characterize -> calibrate -> anomaly -> gradcam -> diagnose -> decide -> process correlate.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
from src.detect import detect
from src.characterize import characterize, detect_pattern
from src.diagnose import diagnose
from src.decide import decide
from src.gradcam import generate_gradcam
from src.anomaly import evaluate_anomaly
from src.process_correlate import correlate_process_parameters

TEST_IMAGES = [
    "scratches_101.jpg",
    "rolled-in_scale_10.jpg",
    "patches_102.jpg",
    "inclusion_102.jpg"
]

def main():
    val_dir = PROJECT_ROOT / "data" / "NEU-DET-final" / "images" / "val"

    print("=" * 85)
    print("END-TO-END PIPELINE VALIDATION (PHASE 6 & 7 COMPLETE INTEGRATION)")
    print("=" * 85)

    for i, fname in enumerate(TEST_IMAGES, 1):
        img_path = val_dir / fname
        print(f"\n--- [Test #{i}] Image: {fname} ---")
        assert img_path.exists(), f"Image {fname} not found"

        # 1. Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 2. Detect
        dets = detect(str(img_path), conf_threshold=0.25)
        print(f"1. Detections found: {len(dets)}")
        assert len(dets) > 0, f"Expected detections for {fname}"

        # 3. Pattern & Characterize
        pattern = detect_pattern(dets)
        fp = characterize(dets[0], (h, w))
        fp["pattern"] = pattern
        print(f"2. Characterized primary defect: {fp['class']} | Severity: {fp['severity']} | Calibrated Conf: {fp['calibrated_confidence']:.2f}")

        # 4. Grad-CAM
        overlay, hm = generate_gradcam(str(img_path))
        print(f"3. Grad-CAM overlay generated: shape {overlay.shape}, max heatmap {hm.max():.2f}")

        # 5. Anomaly Detection
        anom = evaluate_anomaly(str(img_path), yolo_confidence=fp["confidence"])
        print(f"4. Anomaly score: {anom['anomaly_score']:.3f} | Status: {anom['status']}")

        # 6. Metallurgical Diagnosis (LLM + Vector RAG)
        diag = diagnose(fp, k=2)
        print(f"5. Diagnosis Cited Passage: [{diag['cited_passage_id']}] (Sim score: {diag['confidence']:.4f})")
        print(f"   Root Cause: {diag['probable_origin'][:120]}...")
        assert diag["cited_passage_id"] != "NONE", "Diagnosis must cite a valid passage ID"

        # 7. Decision Layer
        decision = decide(fp, diag)
        print(f"6. Recommended Mill Action: [{decision['action']}] via {decision['rule_triggered']}")

        # 8. Process Correlation
        corr = correlate_process_parameters(fp["class"])
        print(f"7. Process Correlation (Simulated): Batch {corr['batch_id']} ({corr['steel_grade']})")

    print("\n" + "=" * 85)
    print("ALL 4 REAL TEST IMAGES PASSED END-TO-END PIPELINE VERIFICATION!")
    print("=" * 85)

if __name__ == "__main__":
    main()
