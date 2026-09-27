"""
Script to fit Temperature Scaling calibration on real NEU-DET validation set images.
Evaluates raw confidence vs IoU ground-truth matching and computes real ECE before and after.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from src.detect import detect
from src.calibrate import TemperatureScaler, compute_ece

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)

def load_ground_truth(label_file: Path, img_size=(200, 200)):
    if not label_file.exists():
        return []
    boxes = []
    w, h = img_size
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = [float(p) for p in parts[1:5]]
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                boxes.append((cls_id, [x1, y1, x2, y2]))
    return boxes

def main():
    val_images_dir = PROJECT_ROOT / "data" / "NEU-DET-final" / "images" / "val"
    val_labels_dir = PROJECT_ROOT / "data" / "NEU-DET-final" / "labels" / "val"

    image_files = sorted(list(val_images_dir.glob("*.jpg")))[:60] # sample 60 images for fast fitting
    print(f"Collecting detection confidence samples on {len(image_files)} validation images...")

    all_confs = []
    all_labels = []

    names_map = {0: "crazing", 1: "inclusion", 2: "patches", 3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}

    for img_path in image_files:
        lbl_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth(lbl_path)
        dets = detect(str(img_path), conf_threshold=0.15)

        for d in dets:
            raw_conf = d["confidence"]
            det_box = d["bbox"]
            det_cls = d["class"]

            # Match with GT: label=1 if IoU >= 0.45 and class matches, else 0
            matched = 0
            for gt_cls_id, gt_box in gt_boxes:
                if names_map.get(gt_cls_id) == det_cls:
                    if box_iou(det_box, gt_box) >= 0.45:
                        matched = 1
                        break
            all_confs.append(raw_conf)
            all_labels.append(matched)

    print(f"Total detections evaluated: {len(all_confs)}")
    if len(all_confs) < 10:
        print("Insufficient detections, using default calibration.")
        return

    confs_arr = np.array(all_confs)
    labels_arr = np.array(all_labels)

    ece_before = compute_ece(confs_arr, labels_arr, num_bins=10)

    scaler = TemperatureScaler()
    fitted_t = scaler.fit_on_val(all_confs, all_labels)
    calibrated_confs = np.array([scaler.calibrate(c) for c in all_confs])
    ece_after = compute_ece(calibrated_confs, labels_arr, num_bins=10)

    print("=" * 60)
    print("CONFIDENCE CALIBRATION (TEMPERATURE SCALING) REPORT")
    print("=" * 60)
    print(f"Fitted Temperature Parameter T: {fitted_t:.4f}")
    print(f"ECE Before Calibration:         {ece_before:.4f} ({ece_before*100:.2f}%)")
    print(f"ECE After Calibration:          {ece_after:.4f} ({ece_after*100:.2f}%)")
    print(f"Calibration Error Reduction:    {max(0.0, (ece_before - ece_after)/max(ece_before, 1e-6)*100):.1f}%")
    print("=" * 60)

    scaler.save()
    print("Saved calibrated model parameters to models/calibration_params.json")

if __name__ == "__main__":
    main()
