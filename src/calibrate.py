"""
Module: src/calibrate.py
Description: Post-hoc confidence calibration for YOLOv8 detector using Temperature Scaling.
             Calculates Expected Calibration Error (ECE) before and after calibration.
Inputs: Validation images and labels from data/NEU-DET-final
Outputs: Calibrated confidence score (float), fitted temperature parameter T
# OWNER: Aman
"""

import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "models" / "calibration_params.json"


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler:
    """
    Temperature Scaling for sigmoid confidence scores.
    logit = log(p / (1 - p))
    p_calibrated = sigmoid(logit / T)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit_on_val(self, confidences: List[float], labels: List[int], lr: float = 0.01, max_iter: int = 200) -> float:
        """
        Fit temperature parameter T using gradient descent minimizing Binary Cross Entropy.
        """
        confs = np.clip(np.array(confidences, dtype=np.float32), 1e-5, 1.0 - 1e-5)
        targets = np.array(labels, dtype=np.float32)

        # Logits
        logits = np.log(confs / (1.0 - confs))

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        # Optimize scalar log(T) to guarantee T > 0
        log_temp = nn.Parameter(torch.zeros(1, requires_grad=True))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            t = torch.exp(log_temp)
            loss = criterion(logits_tensor / t, targets_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).item())
        return self.temperature

    def calibrate(self, conf: float) -> float:
        """Apply fitted temperature scaling to a single raw confidence value."""
        conf = max(1e-5, min(1.0 - 1e-5, float(conf)))
        logit = math.log(conf / (1.0 - conf))
        calibrated_logit = logit / max(self.temperature, 1e-3)
        calibrated_conf = 1.0 / (1.0 + math.exp(-calibrated_logit))
        return float(round(calibrated_conf, 4))

    def save(self, filepath: Path = DEFAULT_CALIBRATION_FILE):
        """Save fitted temperature to disk."""
        os.makedirs(filepath.parent, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    @classmethod
    def load(cls, filepath: Path = DEFAULT_CALIBRATION_FILE) -> "TemperatureScaler":
        """Load fitted temperature from disk, defaulting to T=1.0 if not found."""
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    return cls(temperature=data.get("temperature", 1.0))
            except Exception:
                pass
        return cls(temperature=1.0)


# Cached global scaler
_GLOBAL_SCALER = None

def get_calibrator() -> TemperatureScaler:
    global _GLOBAL_SCALER
    if _GLOBAL_SCALER is None:
        _GLOBAL_SCALER = TemperatureScaler.load()
    return _GLOBAL_SCALER


def calibrate_confidence(raw_conf: float) -> float:
    """Convenience function to calibrate a confidence score using the global scaler."""
    scaler = get_calibrator()
    return scaler.calibrate(raw_conf)


if __name__ == "__main__":
    print("Testing Temperature Scaling module...")
    # Synthetic validation data to verify fitting
    np.random.seed(42)
    synthetic_raw_confs = np.random.uniform(0.3, 0.95, 200).tolist()
    # Simulate overconfident model: actual true label rate is slightly lower than raw conf
    synthetic_labels = [1 if np.random.rand() < (c * 0.8) else 0 for c in synthetic_raw_confs]

    scaler = TemperatureScaler()
    ece_before = compute_ece(np.array(synthetic_raw_confs), np.array(synthetic_labels))
    fitted_t = scaler.fit_on_val(synthetic_raw_confs, synthetic_labels)
    calibrated_confs = [scaler.calibrate(c) for c in synthetic_raw_confs]
    ece_after = compute_ece(np.array(calibrated_confs), np.array(synthetic_labels))

    print(f"Fitted Temperature T: {fitted_t:.4f}")
    print(f"ECE Before Calibration: {ece_before:.4f}")
    print(f"ECE After Calibration:  {ece_after:.4f}")
    scaler.save()
    print("Saved calibration parameters successfully.")
