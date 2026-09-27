"""
Module: src/anomaly.py
Description: Anomaly detection on steel surface images using One-Class SVM on pooled YOLOv8 backbone features.
             Identifies out-of-distribution defects, novel failure modes, and uncataloged flaws.
             Wires low YOLO confidence + high anomaly score -> "Unknown Defect Candidate — needs human review".
Inputs: image (file path or numpy array)
Outputs: dict with {"anomaly_score": float [0-1], "is_anomaly": bool, "status": str, "threshold": float}
# OWNER: Ravi (pending review; built by agent for now)

Architectural Decision Rationale:
---------------------------------
We chose One-Class SVM on Global Average Pooled YOLOv8 backbone features over a from-scratch convolutional autoencoder because:
1. Representation Quality: The YOLOv8 backbone has already learned rich multi-scale filters for surface textures, edges, and scale boundaries.
2. Sample Efficiency: One-Class SVM with an RBF kernel learns a tight support-boundary in high-dimensional embedding space without the risk of autoencoder reconstruction blurriness or overfitting.
3. Deterministic Inference: Fast, stable computation without GPU gradient passes during online inference.
"""

import os
import sys
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
import cv2
import torch
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import get_model, DEFAULT_MODEL_PATH

ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.joblib"
DEFAULT_ANOMALY_THRESHOLD = 0.65


class FeatureExtractor:
    """Extracts global average pooled feature vectors from YOLOv8 backbone."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.yolo = get_model(model_path)
        self.model = self.yolo.model
        self.model.eval()

        self.features = None
        # Hook layer 9 (SPPF) or layer 8 (last C2f)
        self.target_layer = self.model.model[9] if len(self.model.model) > 9 else self.model.model[-2]
        self.target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features = output

    def extract(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Extract a 1D feature vector from image."""
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_input

        resized = cv2.resize(rgb, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            _ = self.model(tensor)

        # Global average pool over spatial dimensions (B, C, H, W) -> (C,)
        if self.features is not None:
            pooled = torch.mean(self.features, dim=(2, 3)).squeeze(0).cpu().numpy()
            return pooled
        else:
            return np.zeros(256, dtype=np.float32)


class AnomalyDetector:
    """One-Class SVM model with feature standardization and normalized anomaly score mapping."""

    def __init__(self, nu: float = 0.08, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(nu=self.nu, kernel="rbf", gamma=self.gamma)
        self.is_fitted = False
        self.extractor = FeatureExtractor()

    def fit(self, feature_matrix: np.ndarray):
        """Fit scaler and One-Class SVM on training feature embeddings."""
        scaled_features = self.scaler.fit_transform(feature_matrix)
        self.svm.fit(scaled_features)
        self.is_fitted = True

    def score(self, image_input: Union[str, Path, np.ndarray]) -> float:
        """
        Compute normalized anomaly score in [0.0, 1.0].
        Higher score = greater deviation from learned distribution.
        """
        if not self.is_fitted:
            return 0.5  # Neutral default if not yet trained

        feat = self.extractor.extract(image_input).reshape(1, -1)
        scaled = self.scaler.transform(feat)
        decision = self.svm.decision_function(scaled)[0]

        # Sigmoid normalization mapping: positive decision = inlier, negative = outlier
        # anomaly_score = 1.0 / (1.0 + exp(decision * 2.0))
        norm_score = 1.0 / (1.0 + np.exp(decision * 2.5))
        return float(round(norm_score, 4))

    def evaluate(
        self,
        image_input: Union[str, Path, np.ndarray],
        yolo_confidence: float = 1.0,
        threshold: float = DEFAULT_ANOMALY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Evaluate image for potential unknown defect candidacy.
        """
        score = self.score(image_input)
        is_anomaly = (score >= threshold) and (yolo_confidence < 0.40)

        if is_anomaly:
            status = "Unknown Defect Candidate — needs human review"
        elif score >= threshold:
            status = "Atypical Surface Texture — elevated anomaly score"
        else:
            status = "Known Distribution — nominal defect profile"

        return {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "status": status,
            "threshold": threshold,
            "yolo_confidence": yolo_confidence
        }

    def save(self, filepath: Path = ANOMALY_MODEL_PATH):
        os.makedirs(filepath.parent, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "svm": self.svm, "is_fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: Path = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        detector = cls()
        if filepath.exists():
            try:
                data = joblib.load(filepath)
                detector.scaler = data["scaler"]
                detector.svm = data["svm"]
                detector.is_fitted = data["is_fitted"]
            except Exception as e:
                print(f"Note loading anomaly model: {e}")
        return detector


# Global cached detector
_GLOBAL_DETECTOR = None

def get_anomaly_detector() -> AnomalyDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = AnomalyDetector.load()
    return _GLOBAL_DETECTOR


def evaluate_anomaly(image_input: Union[str, Path, np.ndarray], yolo_confidence: float = 1.0) -> Dict[str, Any]:
    """Convenience helper to evaluate anomaly score."""
    detector = get_anomaly_detector()
    return detector.evaluate(image_input, yolo_confidence)


if __name__ == "__main__":
    print("Testing Anomaly Detector module...")
    detector = get_anomaly_detector()
    sample_img = PROJECT_ROOT / "data" / "NEU-DET-final" / "images" / "val" / "scratches_101.jpg"
    if sample_img.exists():
        res = detector.evaluate(sample_img, yolo_confidence=0.84)
        print("Anomaly evaluation result:", res)
