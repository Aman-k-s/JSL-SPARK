"""
Module: src/detect.py
Description: Object detection wrapper for YOLOv8 model trained on NEU-DET steel surface defects.
Inputs: image_path (str or Path), optional conf_threshold (float), optional model_path (str)
Outputs: list of dicts with {"class": str, "bbox": [float, float, float, float], "confidence": float}
# OWNER: Aman
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
from ultralytics import YOLO

# Default model path relative to project root
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Global cached model instance
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def get_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> YOLO:
    """Load or return cached YOLO model instance."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    resolved_path = str(Path(model_path).resolve())
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != resolved_path:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Model weights not found at: {resolved_path}")
        _CACHED_MODEL = YOLO(resolved_path)
        _CACHED_MODEL_PATH = resolved_path
    return _CACHED_MODEL


def detect(
    image_path: Union[str, Path],
    conf_threshold: float = 0.25,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 inference on a single image.
    
    Args:
        image_path: Path to target steel surface image.
        conf_threshold: Confidence threshold cutoff for detections.
        model_path: Path to YOLO model weights.
        
    Returns:
        List of detections, each with {"class": str, "bbox": [x1, y1, x2, y2], "confidence": float}.
    """
    img_path_str = str(Path(image_path).resolve())
    if not os.path.exists(img_path_str):
        raise FileNotFoundError(f"Image not found at: {img_path_str}")

    model = get_model(model_path)
    results = model.predict(source=img_path_str, conf=conf_threshold, verbose=False)
    
    detections: List[Dict[str, Any]] = []
    if not results or len(results) == 0:
        return detections

    res = results[0]
    names = res.names  # mapping from class index to class name

    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf.item())
            xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
            
            detections.append({
                "class": cls_name,
                "bbox": xyxy,
                "confidence": round(conf, 4)
            })

    return detections


if __name__ == "__main__":
    import sys
    test_img = Path("data/NEU-DET-final/images/val/crazing_102.jpg")
    if not test_img.exists():
        test_img = Path("data/NEU-DET-final/images/val").glob("*.jpg")
        test_img = next(test_img, None)
    
    if test_img:
        print(f"Testing detect on {test_img}...")
        dets = detect(test_img)
        print(f"Found {len(dets)} detections:")
        for d in dets:
            print(" ", d)
    else:
        print("No test image found.")
