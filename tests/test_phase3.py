"""
Script to test Phase 3: Detection wrapper + Characterization on real NEU-DET validation images.
Evaluates 6-10 real images covering different defect classes.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.detect import detect
from src.characterize import characterize, detect_pattern

def main():
    val_dir = Path("data/NEU-DET-final/images/val")
    test_files = [
        "crazing_102.jpg",
        "inclusion_102.jpg",
        "patches_102.jpg",
        "pitted_surface_1.jpg",
        "rolled-in_scale_10.jpg",
        "scratches_101.jpg",
        "scratches_113.jpg",
        "inclusion_120.jpg",
    ]

    print("=" * 80)
    print("PHASE 3 VERIFICATION — DETECTION & CHARACTERIZATION FINGERPRINT REPORT")
    print("=" * 80)

    for fname in test_files:
        img_path = val_dir / fname
        if not img_path.exists():
            print(f"File {fname} not found, skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {fname}, skipping.")
            continue

        h, w = img.shape[:2]
        dets = detect(str(img_path), conf_threshold=0.20)
        pattern = detect_pattern(dets)

        print(f"\n[Image: {fname}] Dimensions: {w}x{h} | Detections: {len(dets)} | Pattern: {pattern}")
        if not dets:
            print("  (No detections above threshold)")
            continue

        for i, d in enumerate(dets):
            fp = characterize(d, (h, w))
            print(f"  Defect #{i+1}:")
            print(f"    Class: {fp['class']} (conf: {fp['confidence']:.2f})")
            print(f"    BBox: {fp['bbox']} | Aspect Ratio: {fp['aspect_ratio']:.2f} -> {fp['morphology']}")
            print(f"    Centroid: {fp['centroid']} (norm: {fp['normalized_centroid']}) -> {fp['location']}")
            print(f"    Area: {fp['affected_area_pct']:.2f}% | Severity Score: {fp['severity_score']:.2f} -> {fp['severity']}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
