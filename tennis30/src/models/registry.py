"""Model registry for managing and loading all tracking models."""

from pathlib import Path
from typing import Any, Dict, Optional, Type

from tennis30.src.core.court.base_detector import BaseCourtDetector
from tennis30.src.core.pose.base_estimator import BasePoseEstimator
from tennis30.src.core.tracking.ball.base_tracker import BaseBallTracker


class ModelRegistry:
    """Central registry for all Tennis30 models.

    This registry manages:
    - Ball tracking models (TrackNet, YOLO, SAMURAI, Faster R-CNN)
    - Court detection models (ResNet50, Hough transform)
    - Pose estimation models (YOLOv8-Pose, MediaPipe, HRNet, AlphaPose)

    Models are registered using decorators and can be instantiated by name.

    Example:
        >>> @ModelRegistry.register_ball_tracker('tracknet')
        >>> class TrackNetTracker(BaseBallTracker):
        >>>     ...
        >>>
        >>> # Later, instantiate by name:
        >>> tracker = ModelRegistry.get_ball_tracker('tracknet', config)
    """

    _ball_trackers: Dict[str, Type[BaseBallTracker]] = {}
    _court_detectors: Dict[str, Type[BaseCourtDetector]] = {}
    _pose_estimators: Dict[str, Type[BasePoseEstimator]] = {}

    @classmethod
    def register_ball_tracker(cls, name: str):
        """Decorator to register a ball tracking model.

        Args:
            name: Unique identifier for the tracker

        Returns:
            Decorator function

        Example:
            >>> @ModelRegistry.register_ball_tracker('yolo_v1')
            >>> class YOLOTracker(BaseBallTracker):
            >>>     pass
        """

        def decorator(tracker_class: Type[BaseBallTracker]) -> Type[BaseBallTracker]:
            if name in cls._ball_trackers:
                raise ValueError(f"Ball tracker '{name}' already registered")
            cls._ball_trackers[name] = tracker_class
            return tracker_class

        return decorator

    @classmethod
    def register_court_detector(cls, name: str):
        """Decorator to register a court detection model.

        Args:
            name: Unique identifier for the detector

        Returns:
            Decorator function
        """

        def decorator(detector_class: Type[BaseCourtDetector]) -> Type[BaseCourtDetector]:
            if name in cls._court_detectors:
                raise ValueError(f"Court detector '{name}' already registered")
            cls._court_detectors[name] = detector_class
            return detector_class

        return decorator

    @classmethod
    def register_pose_estimator(cls, name: str):
        """Decorator to register a pose estimation model.

        Args:
            name: Unique identifier for the estimator

        Returns:
            Decorator function
        """

        def decorator(estimator_class: Type[BasePoseEstimator]) -> Type[BasePoseEstimator]:
            if name in cls._pose_estimators:
                raise ValueError(f"Pose estimator '{name}' already registered")
            cls._pose_estimators[name] = estimator_class
            return estimator_class

        return decorator

    @classmethod
    def get_ball_tracker(
        cls, name: str, config: Dict[str, Any], device: str = "cuda"
    ) -> BaseBallTracker:
        """Get a ball tracker instance by name.

        Args:
            name: Tracker identifier
            config: Configuration dictionary
            device: Computation device

        Returns:
            Instantiated ball tracker

        Raises:
            ValueError: If tracker name not registered
        """
        if name not in cls._ball_trackers:
            available = ", ".join(cls._ball_trackers.keys())
            raise ValueError(
                f"Unknown ball tracker: '{name}'. Available: {available}"
            )

        tracker_class = cls._ball_trackers[name]
        return tracker_class(config, device)

    @classmethod
    def get_court_detector(
        cls, name: str, config: Dict[str, Any], device: str = "cuda"
    ) -> BaseCourtDetector:
        """Get a court detector instance by name.

        Args:
            name: Detector identifier
            config: Configuration dictionary
            device: Computation device

        Returns:
            Instantiated court detector

        Raises:
            ValueError: If detector name not registered
        """
        if name not in cls._court_detectors:
            available = ", ".join(cls._court_detectors.keys())
            raise ValueError(
                f"Unknown court detector: '{name}'. Available: {available}"
            )

        detector_class = cls._court_detectors[name]
        return detector_class(config, device)

    @classmethod
    def get_pose_estimator(
        cls, name: str, config: Dict[str, Any], device: str = "cuda"
    ) -> BasePoseEstimator:
        """Get a pose estimator instance by name.

        Args:
            name: Estimator identifier
            config: Configuration dictionary
            device: Computation device

        Returns:
            Instantiated pose estimator

        Raises:
            ValueError: If estimator name not registered
        """
        if name not in cls._pose_estimators:
            available = ", ".join(cls._pose_estimators.keys())
            raise ValueError(
                f"Unknown pose estimator: '{name}'. Available: {available}"
            )

        estimator_class = cls._pose_estimators[name]
        return estimator_class(config, device)

    @classmethod
    def list_ball_trackers(cls) -> list[str]:
        """List all registered ball trackers."""
        return list(cls._ball_trackers.keys())

    @classmethod
    def list_court_detectors(cls) -> list[str]:
        """List all registered court detectors."""
        return list(cls._court_detectors.keys())

    @classmethod
    def list_pose_estimators(cls) -> list[str]:
        """List all registered pose estimators."""
        return list(cls._pose_estimators.keys())

    @classmethod
    def list_all(cls) -> Dict[str, list[str]]:
        """List all registered models."""
        return {
            "ball_trackers": cls.list_ball_trackers(),
            "court_detectors": cls.list_court_detectors(),
            "pose_estimators": cls.list_pose_estimators(),
        }


# Model metadata for downloads
MODEL_METADATA = {
    "tracknet": {
        "type": "ball_tracker",
        "description": "TrackNet - Heatmap-based ball detection (VGG-16)",
        "url": "https://github.com/alenzenx/tennis-tracking/releases/download/v1.0/tracknet_best.pth",
        "filename": "tracknet_best.pth",
        "size_mb": 50,
        "checksum": None,  # Add SHA256 checksum
    },
    "yolov8n_tennis_v1": {
        "type": "ball_tracker",
        "description": "YOLOv8 Nano fine-tuned on tennis balls (variant 1)",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # Placeholder
        "filename": "yolov8n_tennis_v1.pt",
        "size_mb": 6,
        "checksum": None,
    },
    "yolov8n_tennis_v2": {
        "type": "ball_tracker",
        "description": "YOLOv8 Nano fine-tuned on tennis balls (variant 2)",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # Placeholder
        "filename": "yolov8n_tennis_v2.pt",
        "size_mb": 6,
        "checksum": None,
    },
    "sam2.1_hiera_large": {
        "type": "ball_tracker",
        "description": "SAMURAI - SAM 2.1 with motion-aware memory",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 800,
        "checksum": None,
    },
    "fasterrcnn_resnet50": {
        "type": "ball_tracker",
        "description": "Faster R-CNN ResNet50 for ball detection",
        "url": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "filename": "fasterrcnn_resnet50.pth",
        "size_mb": 160,
        "checksum": None,
    },
    "yolov8n": {
        "type": "player_tracker",
        "description": "YOLOv8 Nano for player detection",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "filename": "yolov8n.pt",
        "size_mb": 6,
        "checksum": None,
    },
    "yolov8x_pose": {
        "type": "pose_estimator",
        "description": "YOLOv8 X-Large Pose (17 keypoints)",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt",
        "filename": "yolov8x-pose.pt",
        "size_mb": 100,
        "checksum": None,
    },
    "bounce_classifier": {
        "type": "physics",
        "description": "TimeSeriesForest classifier for bounce detection",
        "url": None,  # Included in repo
        "filename": "bounce_classifier.pkl",
        "size_mb": 0.5,
        "checksum": None,
    },
}


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a model.

    Args:
        model_name: Model identifier

    Returns:
        Model metadata dictionary or None if not found
    """
    return MODEL_METADATA.get(model_name)


def list_available_models(model_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """List all available models with metadata.

    Args:
        model_type: Filter by type (ball_tracker, court_detector, pose_estimator, etc.)
                   If None, returns all models

    Returns:
        Dictionary of model_name -> metadata
    """
    if model_type is None:
        return MODEL_METADATA

    return {
        name: meta
        for name, meta in MODEL_METADATA.items()
        if meta["type"] == model_type
    }
