"""Abstract base class for court detection."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseCourtDetector(ABC):
    """Abstract base class for tennis court detection and mapping.

    Defines the interface for court detection algorithms that extract court keypoints
    and compute homography matrices for 2D→3D coordinate transformation.

    Attributes:
        config: Configuration dictionary
        device: Computation device
        keypoints: Detected court keypoints (14 points)
        homography_matrix: 3x3 homography matrix for 2D→3D mapping
    """

    # Standard tennis court dimensions (meters)
    COURT_LENGTH = 23.77  # Full court length
    COURT_WIDTH_SINGLES = 8.23  # Singles court width
    COURT_WIDTH_DOUBLES = 10.97  # Doubles court width
    SERVICE_LINE_DISTANCE = 6.40  # Distance from net to service line
    NET_HEIGHT = 0.914  # Net height at center (meters)

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        """Initialize court detector.

        Args:
            config: Configuration dictionary
            device: Computation device ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.keypoints: Optional[np.ndarray] = None
        self.homography_matrix: Optional[np.ndarray] = None

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect court keypoints in a frame.

        Args:
            frame: Input frame (H, W, 3) in BGR format

        Returns:
            Detection result containing:
                - keypoints: Array of shape (14, 2) with [x, y] coordinates
                - confidence: Detection confidence (0.0-1.0)
                - court_type: 'singles' or 'doubles'

        Keypoint order (14 points):
            0-1:   Baseline left/right (bottom of image)
            2-3:   Baseline left/right (top of image)
            4-5:   Service line left/right (bottom)
            6-7:   Service line left/right (top)
            8-9:   Net posts left/right
            10-11: Sideline bottom points
            12-13: Sideline top points
        """
        pass

    @abstractmethod
    def compute_homography(
        self, keypoints: np.ndarray, court_type: str = "singles"
    ) -> np.ndarray:
        """Compute homography matrix from court keypoints.

        Args:
            keypoints: Court keypoints array (14, 2)
            court_type: 'singles' or 'doubles'

        Returns:
            Homography matrix (3, 3) for 2D→3D transformation
        """
        pass

    def map_to_3d(
        self, points_2d: np.ndarray, homography: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Map 2D pixel coordinates to 3D court coordinates.

        Args:
            points_2d: Array of 2D points (N, 2) in [x, y] format
            homography: Homography matrix (3, 3). If None, uses self.homography_matrix

        Returns:
            Array of 3D court coordinates (N, 3) in meters [x, y, z=0]

        Raises:
            ValueError: If homography is None and self.homography_matrix is not set
        """
        if homography is None:
            if self.homography_matrix is None:
                raise ValueError("Homography matrix not computed. Call detect() first.")
            homography = self.homography_matrix

        # Convert to homogeneous coordinates
        n_points = points_2d.shape[0]
        points_homogeneous = np.hstack([points_2d, np.ones((n_points, 1))])

        # Apply homography
        points_3d_homogeneous = homography @ points_homogeneous.T
        points_3d_homogeneous = points_3d_homogeneous.T

        # Convert back from homogeneous
        points_3d = points_3d_homogeneous[:, :2] / points_3d_homogeneous[:, 2:3]

        # Add z=0 (court plane)
        points_3d = np.hstack([points_3d, np.zeros((n_points, 1))])

        return points_3d

    def map_to_2d(
        self, points_3d: np.ndarray, homography: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Map 3D court coordinates back to 2D pixel coordinates.

        Args:
            points_3d: Array of 3D points (N, 3) in meters
            homography: Homography matrix (3, 3). If None, uses self.homography_matrix

        Returns:
            Array of 2D pixel coordinates (N, 2)
        """
        if homography is None:
            if self.homography_matrix is None:
                raise ValueError("Homography matrix not computed. Call detect() first.")
            homography = self.homography_matrix

        # Use inverse homography
        homography_inv = np.linalg.inv(homography)

        # Take only x, y from 3D points (ignore z)
        points_2d_world = points_3d[:, :2]

        # Convert to homogeneous
        n_points = points_2d_world.shape[0]
        points_homogeneous = np.hstack([points_2d_world, np.ones((n_points, 1))])

        # Apply inverse homography
        points_2d_homogeneous = homography_inv @ points_homogeneous.T
        points_2d_homogeneous = points_2d_homogeneous.T

        # Convert back
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]

        return points_2d

    def get_court_dimensions(self, court_type: str = "singles") -> Dict[str, float]:
        """Get standard court dimensions.

        Args:
            court_type: 'singles' or 'doubles'

        Returns:
            Dictionary with court dimensions in meters
        """
        width = (
            self.COURT_WIDTH_SINGLES
            if court_type == "singles"
            else self.COURT_WIDTH_DOUBLES
        )

        return {
            "length": self.COURT_LENGTH,
            "width": width,
            "service_line_distance": self.SERVICE_LINE_DISTANCE,
            "net_height": self.NET_HEIGHT,
            "baseline_to_net": self.COURT_LENGTH / 2,
        }

    def is_point_in_court(
        self, point_3d: np.ndarray, court_type: str = "singles", margin: float = 0.5
    ) -> bool:
        """Check if a 3D point is within court boundaries.

        Args:
            point_3d: 3D point (3,) in meters [x, y, z]
            court_type: 'singles' or 'doubles'
            margin: Margin outside court to still consider valid (meters)

        Returns:
            True if point is within court boundaries (+ margin)
        """
        dims = self.get_court_dimensions(court_type)

        x, y = point_3d[0], point_3d[1]

        # Check boundaries
        in_x = -margin <= x <= dims["length"] + margin
        in_y = -margin <= y <= dims["width"] + margin

        return in_x and in_y

    def visualize_court(
        self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Draw court lines and keypoints on frame.

        Args:
            frame: Input frame (H, W, 3)
            keypoints: Court keypoints (14, 2). If None, uses self.keypoints

        Returns:
            Frame with court visualization drawn
        """
        import cv2

        if keypoints is None:
            if self.keypoints is None:
                raise ValueError("No keypoints available. Call detect() first.")
            keypoints = self.keypoints

        frame_vis = frame.copy()

        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(frame_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(
                frame_vis,
                str(i),
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Draw court lines (connect keypoints)
        line_pairs = [
            (0, 1),  # Bottom baseline
            (2, 3),  # Top baseline
            (4, 5),  # Bottom service line
            (6, 7),  # Top service line
            (0, 2),  # Left sideline
            (1, 3),  # Right sideline
            (8, 9),  # Net
        ]

        for i, j in line_pairs:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = tuple(map(int, keypoints[i]))
                pt2 = tuple(map(int, keypoints[j]))
                cv2.line(frame_vis, pt1, pt2, (0, 0, 255), 2)

        return frame_vis

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"has_keypoints={self.keypoints is not None})"
        )
