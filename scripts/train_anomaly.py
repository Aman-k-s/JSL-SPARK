"""
Script to train One-Class SVM Anomaly Detector on reference NEU-DET training images.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from src.anomaly import AnomalyDetector, FeatureExtractor

def main():
    train_dir = PROJECT_ROOT / "data" / "NEU-DET-final" / "images" / "train"
    image_files = sorted(list(train_dir.glob("*.jpg")))
    
    # Sample 100 diverse training images across classes for quick and robust manifold fitting
    sample_images = image_files[::max(1, len(image_files) // 100)][:100]
    print(f"Extracting YOLOv8 backbone embeddings for {len(sample_images)} training images...")

    extractor = FeatureExtractor()
    features = []

    for i, img_path in enumerate(sample_images, 1):
        feat = extractor.extract(str(img_path))
        features.append(feat)
        if i % 25 == 0:
            print(f"  Processed {i}/{len(sample_images)} images...")

    feature_matrix = np.array(features)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    detector = AnomalyDetector(nu=0.08)
    detector.fit(feature_matrix)
    detector.save()
    print("One-Class SVM Anomaly Detector trained and saved to models/anomaly_detector.joblib!")

if __name__ == "__main__":
    main()
