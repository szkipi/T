"""YOLO-based ball tracker using Ultralytics YOLOv8."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tennis30.src.core.tracking.ball.base_tracker import BaseBallTracker
from tennis30.src.models.registry import ModelRegistry


@ModelRegistry.register_ball_tracker("yolo_v1")
@ModelRegistry.register_ball_tracker("yolo_v2")
class YOLOBallTracker(BaseBallTracker):
    """YOLOv8-based tennis ball tracker.

    Uses Ultralytics YOLOv8 fine-tuned on tennis ball dataset.
    Supports both v1 and v2 variants (different augmentations/hyperparams).

    Attributes:
        model: YOLO model instance
        input_size: Input image size (width, height)
        batch_size: Batch size for inference
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        """Initialize YOLO tracker.

        Args:
            config: Configuration containing:
                - model_path: Path to YOLOv8 weights (.pt file)
                - input_size: Input size [width, height]
                - batch_size: Batch size for inference
                - weight: Voting weight
                - threshold: Confidence threshold
            device: Computation device
        """
        super().__init__(config, device)

        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.batch_size = config.get("batch_size", 16)
        self.model_path = config.get("model_path")

        # Load model (lazy loading - only when needed)
        if self.model_path:
            self.load_model(self.model_path)

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load YOLO model.

        Args:
            checkpoint_path: Path to .pt file. If None, uses config.model_path
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )

        if checkpoint_path is None:
            checkpoint_path = self.model_path

        if checkpoint_path is None:
            raise ValueError("No model_path specified in config")

        model_path = Path(checkpoint_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}\n"
                f"Download with: python scripts/download_models.py yolo_v1"
            )

        # Load model
        self.model = YOLO(str(model_path))

        # Move to device
        if self.device == "cuda":
            self.model.to("cuda")

        print(f"✓ Loaded YOLO model from {model_path}")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect ball in single frame.

        Args:
            frame: Input frame (H, W, 3) BGR

        Returns:
            Detection dictionary with position, confidence, bbox
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.threshold,
            verbose=False,
            device=self.device,
        )

        # Extract detections
        result = results[0]

        # Check if any detections
        if len(result.boxes) == 0:
            return {
                "position": None,
                "confidence": 0.0,
                "bbox": None,
            }

        # Get the detection with highest confidence
        confidences = result.boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        # Get bounding box (x1, y1, x2, y2)
        bbox = result.boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])

        # Compute center
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        return {
            "position": (float(x_center), float(y_center)),
            "confidence": confidence,
            "bbox": tuple(map(float, bbox)),
        }

    def detect_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Detect ball in multiple frames (batched).

        Args:
            frames: List of input frames
            batch_size: Batch size (overrides config batch_size if provided)

        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch_size = batch_size or self.batch_size
        results = []

        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]

            # Run batch inference
            batch_results = self.model.predict(
                batch,
                imgsz=self.input_size,
                conf=self.threshold,
                verbose=False,
                device=self.device,
            )

            # Extract detections
            for result in batch_results:
                if len(result.boxes) == 0:
                    results.append({
                        "position": None,
                        "confidence": 0.0,
                        "bbox": None,
                    })
                    continue

                # Get best detection
                confidences = result.boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)

                bbox = result.boxes.xyxy[best_idx].cpu().numpy()
                confidence = float(confidences[best_idx])

                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2

                results.append({
                    "position": (float(x_center), float(y_center)),
                    "confidence": confidence,
                    "bbox": tuple(map(float, bbox)),
                })

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"YOLOBallTracker("
            f"weight={self.weight}, "
            f"threshold={self.threshold}, "
            f"input_size={self.input_size}, "
            f"device={self.device})"
        )
