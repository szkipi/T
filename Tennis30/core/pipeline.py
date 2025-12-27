"""
Tennis30 Main Pipeline

Multi-pass precision pipeline orchestrator.
Coordinates all tracking, physics, and export modules.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

from .ball_tracking import PrecisionBallTracker
from .pose_estimation import PrecisionPoseTracker
from .physics import TennisPhysicsEngine
from .court_detection import Court3DMapper
from .game_state import TennisGameStateTracker
from ..utils.preprocessing import VideoPreprocessor
from ..utils.export import UnityExporter, UnrealExporter, CSVExporter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionPipeline:
    """
    Main Tennis30 precision pipeline.

    Multi-pass architecture:
    1. PASS 1: Ensemble detection (5 models)
    2. PASS 2: Temporal refinement (Kalman + context)
    3. PASS 3: Physics validation (trajectory fitting)
    4. PASS 4: Cross-validation (quality check)

    Args:
        config_path: Path to configuration YAML
        mode: 'fast', 'balanced', or 'precision'
        passes: Number of refinement passes (1-4)
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        mode: str = 'precision',
        passes: int = 4,
        device: str = 'cuda'
    ):
        self.mode = mode
        self.passes = passes
        self.device = device

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        logger.info("Initializing Tennis30 pipeline...")

        self.ball_tracker = PrecisionBallTracker(
            config=self.config.get('ball_tracking', {}),
            device=device
        )

        self.pose_tracker = PrecisionPoseTracker(
            config=self.config.get('pose_estimation', {}),
            device=device
        )

        self.physics_engine = TennisPhysicsEngine(
            config=self.config.get('physics', {})
        )

        self.court_mapper = Court3DMapper(
            config=self.config.get('court_detection', {})
        )

        self.game_state_tracker = TennisGameStateTracker(
            config=self.config.get('game_state', {})
        )

        # Preprocessor
        self.preprocessor = VideoPreprocessor()

        # Exporters
        self.unity_exporter = UnityExporter()
        self.unreal_exporter = UnrealExporter()
        self.csv_exporter = CSVExporter()

        logger.info(f"Pipeline initialized in {mode} mode with {passes} passes")


    def process_video(
        self,
        video_path: str,
        output_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        visualize: bool = False
    ) -> Dict:
        """
        Process tennis match video with multi-pass precision pipeline.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            start_frame: Starting frame number
            end_frame: Ending frame number (None = process all)
            visualize: Generate visualization video

        Returns:
            results: Dictionary containing all tracking data
        """

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Mode: {self.mode}, Passes: {self.passes}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # ===== PREPROCESSING =====
        logger.info("Preprocessing video...")
        frames = self.preprocessor.extract_frames(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame
        )

        fps = self.preprocessor.get_fps(video_path)
        total_frames = len(frames)

        logger.info(f"Loaded {total_frames} frames at {fps} FPS")


        # ===== COURT DETECTION & CALIBRATION =====
        logger.info("Detecting court and calibrating 3D mapping...")

        # Use first frame for court detection
        court_keypoints, homography_matrix = self.court_mapper.detect_and_calibrate(
            frames[0]
        )

        court_info = {
            'keypoints_2d': court_keypoints,
            'homography': homography_matrix,
            'dimensions': self.court_mapper.court_dims
        }

        logger.info(f"Court detected with {len(court_keypoints)} keypoints")


        # ===== BALL TRACKING (Multi-Pass) =====
        logger.info(f"Ball tracking with {self.passes}-pass precision...")

        ball_results = self.ball_tracker.track_precision(
            frames=frames,
            court_info=court_info,
            num_passes=self.passes
        )

        logger.info(f"Ball tracking complete. Quality: {ball_results['avg_quality']:.2%}")


        # ===== PLAYER TRACKING & POSE ESTIMATION =====
        logger.info("Player detection and pose estimation...")

        player_results = self.pose_tracker.track_and_pose(
            frames=frames,
            court_keypoints=court_keypoints,
            num_passes=self.passes
        )

        logger.info(f"Detected {len(player_results['players'])} players")


        # ===== PHYSICS VALIDATION & TRAJECTORY PREDICTION =====
        logger.info("Physics validation and trajectory prediction...")

        physics_results = self.physics_engine.process(
            ball_positions=ball_results['positions'],
            player_positions=player_results['players'],
            court_info=court_info
        )

        logger.info(f"Physics validation complete. Bounces detected: {len(physics_results['bounces'])}")


        # ===== GAME STATE TRACKING =====
        logger.info("Tracking game state (rallies, sets, games)...")

        game_state = self.game_state_tracker.track_game_state(
            frames=frames,
            ball_data=physics_results['ball_trajectory'],
            player_data=player_results['players']
        )

        logger.info(f"Game state tracked. Rallies: {len(game_state['rallies'])}")


        # ===== COMPILE RESULTS =====
        results = {
            'metadata': {
                'video_path': video_path,
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps,
                'mode': self.mode,
                'passes': self.passes,
                'quality_score': ball_results['avg_quality'],
                'timestamp': self._get_timestamp()
            },
            'court': court_info,
            'ball': physics_results['ball_trajectory'],
            'players': player_results['players'],
            'game_state': game_state,
            'quality_metrics': {
                'ball_detection_rate': ball_results['detection_rate'],
                'pose_detection_rate': player_results['detection_rate'],
                'interpolation_rate': ball_results['interpolation_rate']
            }
        }


        # ===== EXPORT RESULTS =====
        logger.info("Exporting results...")

        # Save JSON
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Export CSV time series
        self.csv_exporter.export(results, output_dir)

        # Export Unity format
        unity_path = os.path.join(output_dir, 'unity_export.json')
        self.unity_exporter.export(results, unity_path)

        # Export Unreal format
        unreal_path = os.path.join(output_dir, 'unreal_export.json')
        self.unreal_exporter.export(results, unreal_path)

        logger.info(f"✓ Results exported to {output_dir}")


        # ===== VISUALIZATION (Optional) =====
        if visualize:
            logger.info("Generating visualization video...")
            viz_path = os.path.join(output_dir, 'visualization.mp4')
            self._create_visualization(
                frames,
                results,
                viz_path
            )
            logger.info(f"✓ Visualization saved to {viz_path}")


        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Quality Score: {results['metadata']['quality_score']:.2%}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        return results


    def export_unity(self, results: Dict, output_path: str):
        """Export results in Unity-compatible format."""
        self.unity_exporter.export(results, output_path)
        logger.info(f"Unity export saved to {output_path}")


    def export_unreal(self, results: Dict, output_path: str):
        """Export results in Unreal Engine format."""
        self.unreal_exporter.export(results, output_path)
        logger.info(f"Unreal export saved to {output_path}")


    def export_custom(
        self,
        results: Dict,
        output_path: str,
        format: str = 'json',
        **kwargs
    ):
        """Export in custom format."""
        # TODO: Implement custom exporters
        pass


    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../config/precision_config.yaml'
            )

        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}")
            return {}


    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


    def _create_visualization(
        self,
        frames: List[np.ndarray],
        results: Dict,
        output_path: str
    ):
        """
        Create visualization video with overlays.

        Shows:
        - Ball trajectory
        - Player skeletons
        - Court lines
        - Game state
        """
        # TODO: Implement visualization
        pass


if __name__ == '__main__':
    # Example usage
    pipeline = PrecisionPipeline(
        mode='precision',
        passes=4
    )

    results = pipeline.process_video(
        video_path='input/match.mp4',
        output_dir='output/match_001',
        visualize=True
    )
