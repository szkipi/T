"""Abstract base class for pose estimation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class KeypointFormat(Enum):
    """Standard keypoint formats."""

    COCO_17 = "coco_17"  # 17 keypoints (COCO format)
    COCO_WHOLEBODY_133 = "coco_wholebody_133"  # 133 keypoints (full body + hands + face)
    MEDIAPIPE_33 = "mediapipe_33"  # 33 keypoints (MediaPipe)
    ALPHAPOSE_26 = "alphapose_26"  # 26 keypoints (AlphaPose)
    CUSTOM = "custom"  # Custom keypoint format


# COCO 17 keypoint names (standard format)
COCO_17_KEYPOINTS = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]


class BasePoseEstimator(ABC):
    """Abstract base class for human pose estimation.

    Defines interface for pose estimation models that detect keypoints
    (joints) on tennis players for skeleton tracking.

    Attributes:
        config: Configuration dictionary
        device: Computation device
        keypoint_format: Format of output keypoints
        num_keypoints: Number of keypoints detected
        weight: Voting weight for ensemble
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        keypoint_format: KeypointFormat = KeypointFormat.COCO_17,
    ) -> None:
        """Initialize pose estimator.

        Args:
            config: Configuration dictionary
            device: Computation device ('cuda' or 'cpu')
            keypoint_format: Output keypoint format
        """
        self.config = config
        self.device = device
        self.keypoint_format = keypoint_format
        self.num_keypoints = self._get_num_keypoints(keypoint_format)
        self.weight = config.get("weight", 1.0)
        self.threshold = config.get("threshold", 0.3)

    def _get_num_keypoints(self, fmt: KeypointFormat) -> int:
        """Get number of keypoints for a format."""
        mapping = {
            KeypointFormat.COCO_17: 17,
            KeypointFormat.COCO_WHOLEBODY_133: 133,
            KeypointFormat.MEDIAPIPE_33: 33,
            KeypointFormat.ALPHAPOSE_26: 26,
        }
        return mapping.get(fmt, 17)

    @abstractmethod
    def estimate(
        self, frame: np.ndarray, bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[Dict[str, Any]]:
        """Estimate poses for people in frame.

        Args:
            frame: Input frame (H, W, 3) in BGR format
            bboxes: Optional list of bounding boxes (x1, y1, x2, y2) for each person.
                   If None, detector will find people automatically.

        Returns:
            List of pose dictionaries, one per detected person:
                - keypoints: Array (num_keypoints, 2 or 3) with [x, y] or [x, y, confidence]
                - bbox: Bounding box (x1, y1, x2, y2)
                - confidence: Overall pose confidence
                - keypoint_scores: Per-keypoint confidence scores (num_keypoints,)
                - person_id: Optional tracking ID

        Example:
            >>> estimator = YOLOPoseEstimator(config)
            >>> poses = estimator.estimate(frame)
            >>> for pose in poses:
            >>>     print(f"Person at {pose['bbox']} with {len(pose['keypoints'])} keypoints")
        """
        pass

    @abstractmethod
    def estimate_batch(
        self, frames: List[np.ndarray], batch_size: int = 8
    ) -> List[List[Dict[str, Any]]]:
        """Estimate poses for multiple frames (batched).

        Args:
            frames: List of input frames
            batch_size: Number of frames to process at once

        Returns:
            List of pose lists (one per frame)
        """
        pass

    def filter_low_confidence_keypoints(
        self, keypoints: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Filter out keypoints with confidence below threshold.

        Args:
            keypoints: Keypoint array (N, 2) or (N, 3)
            scores: Confidence scores (N,)
            threshold: Confidence threshold. If None, uses self.threshold

        Returns:
            Filtered keypoints with low-confidence points set to NaN
        """
        if threshold is None:
            threshold = self.threshold

        filtered = keypoints.copy()
        low_conf_mask = scores < threshold
        filtered[low_conf_mask] = np.nan

        return filtered

    def apply_temporal_smoothing(
        self, keypoint_history: List[np.ndarray], window_size: int = 5
    ) -> np.ndarray:
        """Apply temporal smoothing to reduce jitter.

        Args:
            keypoint_history: List of keypoint arrays from consecutive frames
            window_size: Size of smoothing window

        Returns:
            Smoothed keypoints for the latest frame
        """
        if len(keypoint_history) < window_size:
            return keypoint_history[-1]

        # Take last window_size frames
        recent = keypoint_history[-window_size:]

        # Simple moving average (ignore NaN values)
        stacked = np.stack(recent, axis=0)  # (window_size, num_keypoints, 2/3)
        smoothed = np.nanmean(stacked, axis=0)

        return smoothed

    def get_skeleton_connections(self) -> List[Tuple[int, int]]:
        """Get pairs of keypoints to connect for skeleton visualization.

        Returns:
            List of (keypoint_i, keypoint_j) pairs to draw as lines
        """
        if self.keypoint_format == KeypointFormat.COCO_17:
            return [
                # Head
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                # Arms
                (5, 6),
                (5, 7),
                (7, 9),
                (6, 8),
                (8, 10),
                # Torso
                (5, 11),
                (6, 12),
                (11, 12),
                # Legs
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16),
            ]
        # Add other formats as needed
        return []

    def visualize_pose(
        self,
        frame: np.ndarray,
        poses: List[Dict[str, Any]],
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
    ) -> np.ndarray:
        """Draw poses on frame.

        Args:
            frame: Input frame (H, W, 3)
            poses: List of pose dictionaries
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoint circles
            draw_skeleton: Whether to draw skeleton lines

        Returns:
            Frame with pose visualization
        """
        import cv2

        frame_vis = frame.copy()

        for pose in poses:
            keypoints = pose["keypoints"]
            bbox = pose.get("bbox")
            scores = pose.get("keypoint_scores")

            # Draw bounding box
            if draw_bbox and bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw skeleton lines
            if draw_skeleton:
                connections = self.get_skeleton_connections()
                for i, j in connections:
                    if i < len(keypoints) and j < len(keypoints):
                        pt1 = keypoints[i]
                        pt2 = keypoints[j]

                        # Skip if either point is NaN
                        if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                            # Check confidence if available
                            if scores is not None:
                                if scores[i] < self.threshold or scores[j] < self.threshold:
                                    continue

                            cv2.line(
                                frame_vis,
                                tuple(map(int, pt1[:2])),
                                tuple(map(int, pt2[:2])),
                                (255, 0, 0),
                                2,
                            )

            # Draw keypoints
            if draw_keypoints:
                for i, kpt in enumerate(keypoints):
                    if not np.isnan(kpt).any():
                        # Check confidence
                        if scores is not None and scores[i] < self.threshold:
                            color = (128, 128, 128)  # Gray for low confidence
                            radius = 3
                        else:
                            color = (0, 0, 255)  # Red for high confidence
                            radius = 5

                        cv2.circle(
                            frame_vis, tuple(map(int, kpt[:2])), radius, color, -1
                        )

        return frame_vis

    def extract_pose_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Extract high-level pose features for analysis.

        Args:
            keypoints: Keypoint array (num_keypoints, 2)

        Returns:
            Dictionary of pose features (e.g., arm angle, body lean)
        """
        # This is a placeholder - implement specific features as needed
        features = {}

        if self.keypoint_format == KeypointFormat.COCO_17:
            # Example: Calculate arm angles
            if len(keypoints) >= 17:
                # Right arm angle (shoulder-elbow-wrist)
                shoulder = keypoints[6]
                elbow = keypoints[8]
                wrist = keypoints[10]

                if not (np.isnan(shoulder).any() or np.isnan(elbow).any() or np.isnan(wrist).any()):
                    v1 = shoulder - elbow
                    v2 = wrist - elbow
                    angle = np.arccos(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    )
                    features["right_arm_angle"] = np.degrees(angle)

        return features

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"format={self.keypoint_format.value}, "
            f"num_keypoints={self.num_keypoints}, "
            f"device={self.device})"
        )
