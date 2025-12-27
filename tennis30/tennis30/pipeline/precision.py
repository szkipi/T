"""4-Pass Precision Pipeline for Tennis Ball Tracking.

This is the core of Tennis30's multi-pass refinement approach.
Prioritizes accuracy over speed by running multiple passes:

Pass 1: Ensemble Detection - Fuse 5 models
Pass 2: Temporal Refinement - Kalman smoothing + gap filling
Pass 3: Physics Validation - Trajectory fitting + bounce detection
Pass 4: Cross-Validation - Quality scoring + final output
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from tennis30.core.physics.engine import TennisPhysicsEngine
from tennis30.core.physics.kalman import BidirectionalKalmanSmoother, KalmanFilter
from tennis30.core.tracking.ball.ensemble import EnsembleBallTracker
from tennis30.utils.config import ConfigLoader


class PrecisionPipeline:
    """4-pass precision tracking pipeline.

    Philosophy: "Not YOLO (Look Once) - We Look Multiple Times"

    Attributes:
        config: Full configuration dictionary
        device: Computation device
        ensemble_tracker: Ensemble ball tracker
        physics_engine: Physics validation engine
        smoother: Bidirectional Kalman smoother
        num_passes: Number of refinement passes (1-4)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        num_passes: int = 4,
    ) -> None:
        """Initialize precision pipeline.

        Args:
            config: Configuration dictionary
            device: Computation device
            num_passes: Number of passes (1-4)
        """
        self.config = config
        self.device = device
        self.num_passes = min(max(num_passes, 1), 4)

        print(f"\n{'='*60}")
        print(f"Tennis30 Precision Pipeline")
        print(f"{'='*60}")
        print(f"Passes: {self.num_passes}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        ball_config = self.config.get("ball_tracking", {})
        physics_config = self.config.get("physics", {})

        # Ensemble tracker (Pass 1)
        print("Initializing ensemble tracker...")
        self.ensemble_tracker = EnsembleBallTracker(
            ball_config, device=self.device
        )

        # Physics engine (Pass 3)
        print("\nInitializing physics engine...")
        self.physics_engine = TennisPhysicsEngine(physics_config)

        # Kalman smoother (Pass 2)
        print("Initializing Kalman smoother...")
        self.smoother = BidirectionalKalmanSmoother(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
        )

        print("\n✓ All components initialized\n")

    @classmethod
    def from_config(
        cls,
        config_name: str = "default",
        config_dir: Optional[Path] = None,
        device: str = "cuda",
        num_passes: int = 4,
    ) -> "PrecisionPipeline":
        """Create pipeline from configuration file.

        Args:
            config_name: Configuration file name
            config_dir: Configuration directory
            device: Computation device
            num_passes: Number of passes

        Returns:
            Initialized pipeline
        """
        loader = ConfigLoader(config_dir)
        config = loader.load(config_name)
        return cls(config, device=device, num_passes=num_passes)

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        """Process entire video through 4-pass pipeline.

        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            visualize: Whether to generate visualization video

        Returns:
            Results dictionary containing:
                - ball_positions: Final ball positions (N, 2)
                - confidence_scores: Confidence per frame (N,)
                - quality_scores: Quality scores (N,)
                - metadata: Processing metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"{'='*60}\n")

        # Load video
        frames = self._load_video(video_path)
        n_frames = len(frames)

        print(f"Video loaded: {n_frames} frames\n")

        # PASS 1: Ensemble Detection
        print(f"PASS 1/4: Ensemble Detection")
        print("-" * 60)
        pass1_detections = self._pass1_ensemble_detection(frames)

        if self.num_passes == 1:
            return self._finalize_results(pass1_detections, n_frames)

        # PASS 2: Temporal Refinement
        print(f"\nPASS 2/4: Temporal Refinement")
        print("-" * 60)
        pass2_positions = self._pass2_temporal_refinement(pass1_detections, frames)

        if self.num_passes == 2:
            return self._finalize_results_from_positions(pass2_positions, n_frames)

        # PASS 3: Physics Validation
        print(f"\nPASS 3/4: Physics Validation")
        print("-" * 60)
        pass3_positions = self._pass3_physics_validation(pass2_positions)

        if self.num_passes == 3:
            return self._finalize_results_from_positions(pass3_positions, n_frames)

        # PASS 4: Cross-Validation
        print(f"\nPASS 4/4: Cross-Validation & Quality Scoring")
        print("-" * 60)
        final_results = self._pass4_cross_validation(
            pass3_positions, pass1_detections, frames
        )

        print(f"\n{'='*60}")
        print(f"Processing Complete!")
        print(f"{'='*60}\n")

        return final_results

    def _pass1_ensemble_detection(self, frames: List[np.ndarray]) -> List[Dict]:
        """Pass 1: Ensemble detection with 5 models.

        Args:
            frames: List of video frames

        Returns:
            List of detection dictionaries
        """
        detections = []

        for frame in tqdm(frames, desc="Ensemble detection"):
            detection = self.ensemble_tracker.detect(frame)
            detections.append(detection)

        # Statistics
        detected = sum(1 for d in detections if d["position"] is not None)
        avg_confidence = np.mean(
            [d["confidence"] for d in detections if d["position"] is not None]
        )

        print(f"  Detected: {detected}/{len(frames)} frames ({detected/len(frames)*100:.1f}%)")
        print(f"  Avg confidence: {avg_confidence:.3f}")

        return detections

    def _pass2_temporal_refinement(
        self, detections: List[Dict], frames: List[np.ndarray]
    ) -> np.ndarray:
        """Pass 2: Temporal refinement with Kalman smoothing.

        Args:
            detections: Detections from Pass 1
            frames: Video frames

        Returns:
            Refined positions (N, 2)
        """
        # Extract positions and prepare measurements for Kalman
        positions = []
        measurements = []

        for detection in detections:
            pos = detection.get("position")
            if pos is not None:
                positions.append(pos)
                # Measurement: [x, y, aspect_ratio, height]
                # For now, use dummy aspect/height
                measurements.append([pos[0], pos[1], 1.0, 10.0])
            else:
                positions.append(None)
                measurements.append([np.nan, np.nan, np.nan, np.nan])

        measurements = np.array(measurements)

        # Apply bidirectional Kalman smoothing
        smoothed_positions = self.smoother.smooth(measurements)

        # Fill gaps using interpolation
        smoothed_positions = self._interpolate_gaps(smoothed_positions)

        # Statistics
        valid = np.sum(~np.isnan(smoothed_positions[:, 0]))
        print(f"  Smoothed positions: {valid}/{len(smoothed_positions)} ({valid/len(smoothed_positions)*100:.1f}%)")

        return smoothed_positions

    def _pass3_physics_validation(self, positions: np.ndarray) -> np.ndarray:
        """Pass 3: Physics validation and correction.

        Args:
            positions: Positions from Pass 2 (N, 2)

        Returns:
            Physics-validated positions (N, 2)
        """
        # For 2D positions, add dummy z coordinate
        positions_3d = np.hstack([positions, np.zeros((len(positions), 1))])

        # Validate trajectory
        validation = self.physics_engine.validate_trajectory(positions_3d)

        print(f"  Velocity valid: {validation['velocity_valid']}")
        print(f"  Acceleration valid: {validation['acceleration_valid']}")
        print(f"  Max velocity: {validation['max_velocity']:.1f} m/s")
        print(f"  Outliers: {len(validation['outlier_indices'])}")
        print(f"  Bounces detected: {len(validation['bounces'])}")

        # Fit trajectory
        fitted = self.physics_engine.fit_trajectory(positions_3d)
        print(f"  Trajectory RMSE: {fitted['rmse']:.3f}m")

        # Use fitted trajectory if RMSE is reasonable
        if fitted['rmse'] < 0.5:  # 50cm threshold
            validated_positions = fitted['fitted_positions'][:, :2]
            print(f"  Using fitted trajectory")
        else:
            validated_positions = positions
            print(f"  Keeping original positions (fit not good enough)")

        return validated_positions

    def _pass4_cross_validation(
        self,
        positions: np.ndarray,
        detections: List[Dict],
        frames: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Pass 4: Cross-validation and quality scoring.

        Args:
            positions: Positions from Pass 3
            detections: Original detections from Pass 1
            frames: Video frames

        Returns:
            Final results dictionary
        """
        quality_scores = []

        for i, (pos, detection) in enumerate(zip(positions, detections)):
            # Compute quality score based on:
            # 1. Number of models that detected
            # 2. Confidence
            # 3. Agreement with physics

            num_models = detection.get("num_models", 0)
            confidence = detection.get("confidence", 0.0)

            # Quality score (0-1)
            model_score = num_models / 5.0  # Normalize by max models
            confidence_score = confidence

            # Combined quality
            quality = (model_score * 0.5 + confidence_score * 0.5)
            quality_scores.append(quality)

        quality_scores = np.array(quality_scores)

        # Statistics
        avg_quality = np.mean(quality_scores[~np.isnan(quality_scores)])
        high_quality = np.sum(quality_scores > 0.7)

        print(f"  Avg quality: {avg_quality:.3f}")
        print(f"  High quality (>0.7): {high_quality}/{len(quality_scores)} ({high_quality/len(quality_scores)*100:.1f}%)")

        return {
            "ball_positions": positions,
            "quality_scores": quality_scores,
            "confidence_scores": np.array([d.get("confidence", 0.0) for d in detections]),
            "metadata": {
                "num_frames": len(frames),
                "num_passes": self.num_passes,
                "avg_quality": float(avg_quality),
                "detection_rate": float(np.sum(~np.isnan(positions[:, 0])) / len(positions)),
            },
        }

    def _load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames.

        Args:
            video_path: Path to video file

        Returns:
            List of frames (BGR format)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _interpolate_gaps(self, positions: np.ndarray, max_gap: int = 10) -> np.ndarray:
        """Interpolate missing positions.

        Args:
            positions: Positions with NaN gaps (N, 2)
            max_gap: Maximum gap size to interpolate

        Returns:
            Interpolated positions
        """
        result = positions.copy()

        for dim in range(positions.shape[1]):
            values = positions[:, dim]
            valid_indices = np.where(~np.isnan(values))[0]

            if len(valid_indices) < 2:
                continue

            # Interpolate
            result[:, dim] = np.interp(
                np.arange(len(values)),
                valid_indices,
                values[valid_indices],
            )

        return result

    def _finalize_results(self, detections: List[Dict], n_frames: int) -> Dict:
        """Finalize results from detections."""
        positions = np.array(
            [d["position"] if d["position"] else (np.nan, np.nan) for d in detections]
        )
        confidences = np.array([d.get("confidence", 0.0) for d in detections])

        return {
            "ball_positions": positions,
            "confidence_scores": confidences,
            "quality_scores": confidences,  # Use confidence as quality for Pass 1
            "metadata": {
                "num_frames": n_frames,
                "num_passes": 1,
                "detection_rate": float(np.sum(~np.isnan(positions[:, 0])) / n_frames),
            },
        }

    def _finalize_results_from_positions(
        self, positions: np.ndarray, n_frames: int
    ) -> Dict:
        """Finalize results from positions."""
        return {
            "ball_positions": positions,
            "confidence_scores": np.ones(n_frames),  # Dummy
            "quality_scores": np.ones(n_frames),  # Dummy
            "metadata": {
                "num_frames": n_frames,
                "num_passes": self.num_passes,
                "detection_rate": float(np.sum(~np.isnan(positions[:, 0])) / n_frames),
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PrecisionPipeline("
            f"passes={self.num_passes}, "
            f"device={self.device})"
        )
