"""
Module: src/gradcam.py
Description: Generates Grad-CAM / Activation heatmaps on the YOLOv8 backbone layers for visual explainability.
             Highlights the spatial surface regions driving the neural network defect classification.
Inputs: image (path or RGB numpy array), optional target_class_id (int), optional model
Outputs: Tuple of (cam_overlay_rgb, raw_heatmap) as uint8 numpy arrays
# OWNER: Aman
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Union, Optional
from src.detect import get_model, DEFAULT_MODEL_PATH


class YOLOBackboneCAM:
    """
    Activation and Gradient Class Activation Mapping (Grad-CAM)
    for YOLOv8 PyTorch model backbone.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.yolo = get_model(model_path)
        self.model = self.yolo.model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Target the deepest representation layer of the YOLOv8 backbone (layer 9: SPPF)
        # or layer 8 (last C2f block)
        self.target_layer = None
        for i, layer in enumerate(self.model.model):
            # Layer 9 is typically SPPF in YOLOv8n
            if i == 9:
                self.target_layer = layer
                break
        if self.target_layer is None:
            self.target_layer = self.model.model[-2]

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        image_input: Union[str, Path, np.ndarray],
        target_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmap overlay.

        Args:
            image_input: Path to image or RGB/BGR numpy array.
            target_size: YOLO input resolution (640, 640).

        Returns:
            Tuple of (overlay_rgb, heatmap_normalized):
                - overlay_rgb: Blended RGB image with jet heatmap overlay
                - heatmap_normalized: 2D float array in [0.0, 1.0]
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            orig_bgr = cv2.imread(str(image_input))
            if orig_bgr is None:
                raise ValueError(f"Could not load image from {image_input}")
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            if len(image_input.shape) == 2:
                orig_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                orig_rgb = image_input.copy()
            else:
                orig_rgb = image_input[:, :, :3]

        orig_h, orig_w = orig_rgb.shape[:2]

        # Preprocess for YOLO
        resized = cv2.resize(orig_rgb, target_size)
        tensor_img = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_img.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        preds = self.model(tensor_img)

        # Handle predictions to extract target loss scalar
        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[0]
        else:
            pred_tensor = preds

        # Target top prediction activation
        if self.activations is not None:
            # If gradients can be computed
            try:
                target_score = pred_tensor.max()
                target_score.backward(retain_graph=False)
                grads = self.gradients.detach().cpu().numpy()[0]  # (C, H, W)
                acts = self.activations.detach().cpu().numpy()[0]   # (C, H, W)
                weights = np.mean(grads, axis=(1, 2), keepdims=True)
                cam = np.sum(weights * acts, axis=0)
                cam = np.maximum(cam, 0)  # ReLU
            except Exception:
                # EigenCAM / Principal Feature Activation Fallback
                acts = self.activations.detach().cpu().numpy()[0]
                cam = np.mean(acts, axis=0)
                cam = np.maximum(cam, 0)
        else:
            cam = np.ones((target_size[1] // 32, target_size[0] // 32))

        # Normalize CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam)

        # Resize heatmap back to original image size
        heatmap_resized = cv2.resize(cam_norm, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Colorize with JET colormap
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend (0.55 original + 0.45 heatmap)
        blended = cv2.addWeighted(orig_rgb, 0.55, heatmap_color_rgb, 0.45, 0)

        return blended, heatmap_resized


# Module-level cached instance
_CAM_INSTANCE = None

def generate_gradcam(image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to generate Grad-CAM overlay."""
    global _CAM_INSTANCE
    if _CAM_INSTANCE is None:
        _CAM_INSTANCE = YOLOBackboneCAM()
    return _CAM_INSTANCE.generate_heatmap(image_input)


if __name__ == "__main__":
    test_img = Path("data/NEU-DET-final/images/val/scratches_101.jpg")
    if test_img.exists():
        print(f"Testing Grad-CAM generation on {test_img}...")
        overlay, hm = generate_gradcam(test_img)
        print(f"Grad-CAM successfully generated! Shape: {overlay.shape}, Heatmap max: {hm.max():.3f}")
