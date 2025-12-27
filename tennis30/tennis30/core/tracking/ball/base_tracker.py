"""Abstract base class for ball trackers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseBallTracker(ABC):
    """Abstract base class for tennis ball tracking models.

    This class defines the interface that all ball tracking models must implement.
    It enables a pluggable architecture where different tracking algorithms can be
    easily swapped and combined in an ensemble.

    Attributes:
        config: Configuration dictionary for the tracker
        weight: Voting weight in ensemble (0.0-1.0)
        threshold: Confidence threshold for detections
        device: Computation device ('cuda' or 'cpu')
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        """Initialize the ball tracker.

        Args:
            config: Configuration dictionary containing:
                - weight: Voting weight (default: 1.0)
                - threshold: Confidence threshold (default: 0.5)
                - model_path: Path to model weights (optional)
            device: Computation device ('cuda' or 'cpu')
        """
        self.config = config
        self.weight = config.get("weight", 1.0)
        self.threshold = config.get("threshold", 0.5)
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load pretrained model weights.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses config.model_path
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect ball in a single frame.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format

        Returns:
            Detection dictionary containing:
                - position: (x, y) tuple of ball center
                - confidence: Detection confidence score (0.0-1.0)
                - bbox: Optional (x1, y1, x2, y2) bounding box
                - mask: Optional binary mask (H, W)
                - heatmap: Optional probability heatmap (H, W)

        Example:
            >>> tracker = YOLOTracker(config)
            >>> result = tracker.detect(frame)
            >>> print(result)
            {
                'position': (512, 384),
                'confidence': 0.95,
                'bbox': (500, 370, 524, 398)
            }
        """
        pass

    @abstractmethod
    def detect_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Detect ball in multiple frames (batched for efficiency).

        Args:
            frames: List of input frames
            batch_size: Number of frames to process at once

        Returns:
            List of detection dictionaries (one per frame)
        """
        pass

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before detection.

        Default implementation returns frame unchanged. Override for model-specific
        preprocessing (resizing, normalization, etc.).

        Args:
            frame: Input frame (H, W, 3)

        Returns:
            Preprocessed frame
        """
        return frame

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """Postprocess model output to standard format.

        Args:
            raw_output: Raw model output (model-specific format)

        Returns:
            Standardized detection dictionary
        """
        raise NotImplementedError("Subclass must implement postprocess()")

    def get_weight(self) -> float:
        """Get voting weight for ensemble.

        Returns:
            Weight value (0.0-1.0)
        """
        return self.weight

    def set_weight(self, weight: float) -> None:
        """Set voting weight for ensemble.

        Args:
            weight: New weight value (0.0-1.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {weight}")
        self.weight = weight

    def is_valid_detection(self, detection: Dict[str, Any]) -> bool:
        """Check if detection meets confidence threshold.

        Args:
            detection: Detection dictionary

        Returns:
            True if confidence >= threshold
        """
        return detection.get("confidence", 0.0) >= self.threshold

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"weight={self.weight}, "
            f"threshold={self.threshold}, "
            f"device={self.device})"
        )


class DummyBallTracker(BaseBallTracker):
    """Dummy tracker for testing purposes.

    Always returns a fixed position with configurable confidence.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        fixed_position: Tuple[int, int] = (512, 384),
        fixed_confidence: float = 0.5,
    ) -> None:
        """Initialize dummy tracker.

        Args:
            config: Configuration dictionary
            device: Computation device
            fixed_position: Position to always return
            fixed_confidence: Confidence to always return
        """
        super().__init__(config, device)
        self.fixed_position = fixed_position
        self.fixed_confidence = fixed_confidence

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """No-op for dummy tracker."""
        pass

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Return fixed detection."""
        return {
            "position": self.fixed_position,
            "confidence": self.fixed_confidence,
            "bbox": None,
        }

    def detect_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Return fixed detection for all frames."""
        return [self.detect(frame) for frame in frames]
