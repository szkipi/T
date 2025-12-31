"""TrackNet-based heatmap ball tracker - Full PyTorch Implementation.

TrackNet uses a VGG-16 encoder-decoder architecture to generate
heatmaps for ball detection, which is particularly effective for
motion-blurred fast-moving balls.

This is the complete implementation using PyTorch.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from tennis30.core.tracking.ball.base_tracker import BaseBallTracker
from tennis30.core.tracking.ball.models.tracknet import create_tracknet
from tennis30.core.tracking.ball.temporal_handler import TemporalFrameHandler
from tennis30.models.registry import ModelRegistry


@ModelRegistry.register_ball_tracker("tracknet")
class TrackNetTracker(BaseBallTracker):
    """TrackNet heatmap-based ball tracker.

    Uses VGG-16 encoder-decoder to generate Gaussian heatmaps.
    Excellent for motion blur and fast balls.

    Attributes:
        model: TrackNet PyTorch model
        temporal_handler: Handles 3-frame temporal input
        input_size: Input size (512, 512)
        heatmap_threshold: Threshold for heatmap peak detection
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        """Initialize TrackNet tracker.

        Args:
            config: Configuration containing:
                - model_path: Path to TrackNet weights (.pth file)
                - input_size: Input size [width, height]
                - threshold: Confidence threshold
                - weight: Voting weight
                - heatmap_threshold: Heatmap peak threshold
            device: Computation device
        """
        super().__init__(config, device)

        self.input_size = tuple(config.get("input_size", [512, 512]))
        self.batch_size = config.get("batch_size", 8)
        self.heatmap_threshold = config.get("heatmap_threshold", 0.5)
        self.model_path = config.get("model_path")
        self.use_temporal = config.get("use_temporal", True)
        self.num_frames = 3 if self.use_temporal else 1

        # Initialize model
        self.model = None
        self.use_fallback = True  # Start with fallback

        # Temporal frame handler
        if self.use_temporal:
            self.temporal_handler = TemporalFrameHandler(
                num_frames=self.num_frames,
                input_size=self.input_size,
            )
        else:
            self.temporal_handler = None

        # Try to load model
        if self.model_path:
            self.load_model(self.model_path)

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load TrackNet model.

        Args:
            checkpoint_path: Path to .pth file
        """
        if checkpoint_path is None:
            checkpoint_path = self.model_path

        if checkpoint_path is None:
            print("⚠ TrackNet: No model_path specified")
            print("  Using fallback color-based detection")
            self.use_fallback = True
            return

        model_path = Path(checkpoint_path)
        if not model_path.exists():
            print(f"⚠ TrackNet model not found: {model_path}")
            print(f"  Download with: python scripts/download_models.py tracknet")
            print(f"  Using fallback detection for now.")
            self.use_fallback = True
            return

        try:
            # Create model
            self.model = create_tracknet(
                input_size=self.input_size,
                num_frames=self.num_frames,
                pretrained=True,
                weights_path=str(model_path),
            )

            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            self.use_fallback = False
            print(f"✓ TrackNet model loaded successfully")
            print(f"  Input: {self.num_frames} frames, {self.input_size}")
            print(f"  Device: {self.device}")

        except Exception as e:
            print(f"⚠ Failed to load TrackNet model: {e}")
            print(f"  Using fallback detection")
            self.use_fallback = True

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect ball using heatmap prediction.

        Args:
            frame: Input frame (H, W, 3) BGR

        Returns:
            Detection with position, confidence, heatmap
        """
        # Use fallback if model not loaded
        if self.use_fallback or self.model is None:
            return self._fallback_detect(frame)

        # Temporal mode: need 3 frames
        if self.use_temporal:
            # Add frame to buffer
            self.temporal_handler.add_frame(frame)

            # Not ready yet
            if not self.temporal_handler.is_ready():
                return {
                    "position": None,
                    "confidence": 0.0,
                    "bbox": None,
                    "heatmap": None,
                }

            # Get temporal input
            input_tensor = self.temporal_handler.get_temporal_input(
                device=self.device,
                normalize=True,
            )

        # Single frame mode
        else:
            # Resize and preprocess
            resized = cv2.resize(frame, self.input_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            input_tensor = torch.from_numpy(rgb).float()
            input_tensor = input_tensor.permute(2, 0, 1)  # HWC -> CHW
            input_tensor = input_tensor / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Model inference
        with torch.no_grad():
            heatmap = self.model(input_tensor)

        # Convert to numpy
        heatmap_np = heatmap.squeeze().cpu().numpy()  # (H, W)

        # Find ball position from heatmap
        position = self.heatmap_to_coords(heatmap_np)

        if position is None:
            return {
                "position": None,
                "confidence": 0.0,
                "bbox": None,
                "heatmap": heatmap_np,
            }

        # Scale position back to original frame size
        scale_x = frame.shape[1] / self.input_size[0]
        scale_y = frame.shape[0] / self.input_size[1]

        x_orig = position[0] * scale_x
        y_orig = position[1] * scale_y

        # Confidence is the peak value in heatmap
        confidence = float(heatmap_np.max())

        return {
            "position": (float(x_orig), float(y_orig)),
            "confidence": confidence,
            "bbox": None,
            "heatmap": heatmap_np,
        }

    def _fallback_detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Fallback detection using color thresholding.

        This is used when the PyTorch model is not available.

        Args:
            frame: Input frame

        Returns:
            Detection dictionary
        """
        # Resize to input size
        resized = cv2.resize(frame, self.input_size)

        # Convert to HSV for yellow ball detection
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Yellow color range (tennis ball)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "position": None,
                "confidence": 0.0,
                "bbox": None,
                "heatmap": mask / 255.0,
            }

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Check minimum area
        if area < 10:
            return {
                "position": None,
                "confidence": 0.0,
                "bbox": None,
                "heatmap": mask / 255.0,
            }

        # Get centroid
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return {
                "position": None,
                "confidence": 0.0,
                "bbox": None,
                "heatmap": mask / 255.0,
            }

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Scale back to original resolution
        scale_x = frame.shape[1] / self.input_size[0]
        scale_y = frame.shape[0] / self.input_size[1]

        cx_orig = cx * scale_x
        cy_orig = cy * scale_y

        # Confidence based on contour area (normalized)
        confidence = min(area / 100.0, 1.0)

        return {
            "position": (float(cx_orig), float(cy_orig)),
            "confidence": float(confidence),
            "bbox": None,
            "heatmap": mask / 255.0,
        }

    def detect_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Detect ball in multiple frames.

        Args:
            frames: List of input frames
            batch_size: Batch size

        Returns:
            List of detections
        """
        # Use fallback for batch processing if model not loaded
        if self.use_fallback or self.model is None:
            return [self.detect(frame) for frame in frames]

        # TODO: Implement optimized batched inference
        # For now, process sequentially
        return [self.detect(frame) for frame in frames]

    def heatmap_to_coords(self, heatmap: np.ndarray) -> Optional[tuple]:
        """Convert heatmap to (x, y) coordinates.

        Finds the peak in the heatmap and returns its coordinates.

        Args:
            heatmap: 2D heatmap array (H, W)

        Returns:
            (x, y) coordinates of peak, or None if below threshold
        """
        # Find maximum value location
        max_val = heatmap.max()

        if max_val < self.heatmap_threshold:
            return None

        # Get coordinates of maximum
        max_loc = np.unravel_index(heatmap.argmax(), heatmap.shape)
        y, x = max_loc  # Note: numpy uses (row, col)

        return (float(x), float(y))

    def reset(self) -> None:
        """Reset temporal buffer."""
        if self.temporal_handler:
            self.temporal_handler.reset()

    def __repr__(self) -> str:
        """String representation."""
        mode = "FALLBACK" if self.use_fallback else "PYTORCH"
        return (
            f"TrackNetTracker("
            f"mode={mode}, "
            f"weight={self.weight}, "
            f"threshold={self.threshold}, "
            f"input_size={self.input_size}, "
            f"temporal={self.use_temporal}, "
            f"device={self.device})"
        )
