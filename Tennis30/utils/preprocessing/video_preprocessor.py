"""Video preprocessing utilities"""

import cv2
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """Preprocess video for tracking pipeline."""

    def extract_frames(self, video_path: str, start_frame: int = 0,
                      end_frame: int = None) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id >= start_frame:
                if end_frame is None or frame_id < end_frame:
                    frames.append(frame)
                else:
                    break

            frame_id += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames

    def get_fps(self, video_path: str) -> float:
        """Get video FPS."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
