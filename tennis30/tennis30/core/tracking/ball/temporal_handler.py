"""Temporal frame handler for multi-frame ball tracking.

Manages frame buffers and creates temporal inputs for models
that require multiple consecutive frames (e.g., TrackNet).
"""

from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import torch


class TemporalFrameHandler:
    """Handles temporal frame buffering for multi-frame tracking.

    Maintains a sliding window of frames and provides methods to:
    - Add new frames to buffer
    - Retrieve N consecutive frames
    - Preprocess frames for model input
    - Convert to PyTorch tensors

    Attributes:
        num_frames: Number of frames in temporal window
        buffer: Frame buffer (deque)
        input_size: Target input size (height, width)
    """

    def __init__(
        self,
        num_frames: int = 3,
        input_size: tuple = (512, 512),
    ):
        """Initialize temporal frame handler.

        Args:
            num_frames: Number of consecutive frames to track
            input_size: (height, width) for resizing
        """
        self.num_frames = num_frames
        self.input_size = input_size
        self.buffer = deque(maxlen=num_frames)

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a new frame to the buffer.

        Args:
            frame: BGR image (H, W, 3)
        """
        # Resize to input size
        resized = cv2.resize(frame, self.input_size)
        self.buffer.append(resized)

    def is_ready(self) -> bool:
        """Check if buffer has enough frames.

        Returns:
            True if buffer has num_frames frames
        """
        return len(self.buffer) == self.num_frames

    def get_frames(self) -> Optional[List[np.ndarray]]:
        """Get current temporal window of frames.

        Returns:
            List of frames, or None if not ready
        """
        if not self.is_ready():
            return None
        return list(self.buffer)

    def get_temporal_input(
        self,
        device: str = "cpu",
        normalize: bool = True,
    ) -> Optional[torch.Tensor]:
        """Get temporal input tensor for model.

        Creates a tensor by concatenating frames along channel dimension.
        For 3 RGB frames: (3, H, W, 3) → (1, 9, H, W)

        Args:
            device: Target device ('cpu' or 'cuda')
            normalize: Whether to normalize to [0, 1]

        Returns:
            Tensor (1, C, H, W) where C = num_frames * 3
            Returns None if buffer not ready
        """
        frames = self.get_frames()
        if frames is None:
            return None

        # Stack frames and convert BGR to RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        # Convert to numpy array: (num_frames, H, W, 3)
        frames_array = np.stack(frames_rgb, axis=0)

        # Transpose to (num_frames, 3, H, W)
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))

        # Reshape to (num_frames * 3, H, W)
        temporal_input = frames_array.reshape(
            self.num_frames * 3,
            self.input_size[0],
            self.input_size[1]
        )

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(temporal_input).float()

        if normalize:
            tensor = tensor / 255.0

        # Add batch dimension: (1, C, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor.to(device)

    def reset(self) -> None:
        """Clear the frame buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemporalFrameHandler("
            f"num_frames={self.num_frames}, "
            f"buffer_size={len(self.buffer)}/{self.num_frames}, "
            f"input_size={self.input_size})"
        )


class BatchTemporalHandler:
    """Batch version of temporal handler for processing multiple videos.

    Manages multiple temporal frame handlers in parallel.
    """

    def __init__(
        self,
        batch_size: int,
        num_frames: int = 3,
        input_size: tuple = (512, 512),
    ):
        """Initialize batch temporal handler.

        Args:
            batch_size: Number of parallel streams
            num_frames: Frames per temporal window
            input_size: Target input size
        """
        self.batch_size = batch_size
        self.handlers = [
            TemporalFrameHandler(num_frames, input_size)
            for _ in range(batch_size)
        ]

    def add_frames(self, frames: List[np.ndarray]) -> None:
        """Add frames to each handler.

        Args:
            frames: List of frames (one per handler)
        """
        assert len(frames) == self.batch_size
        for handler, frame in zip(self.handlers, frames):
            handler.add_frame(frame)

    def is_ready(self) -> bool:
        """Check if all handlers are ready."""
        return all(h.is_ready() for h in self.handlers)

    def get_batch_input(
        self,
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """Get batched temporal input.

        Returns:
            Tensor (B, C, H, W) where B = batch_size
            Returns None if not all handlers ready
        """
        if not self.is_ready():
            return None

        tensors = [h.get_temporal_input(device) for h in self.handlers]
        return torch.cat(tensors, dim=0)

    def reset_all(self) -> None:
        """Reset all handlers."""
        for handler in self.handlers:
            handler.reset()


if __name__ == "__main__":
    # Test temporal handler
    print("🧪 Testing TemporalFrameHandler\n")

    handler = TemporalFrameHandler(num_frames=3, input_size=(512, 512))
    print(handler)

    # Simulate adding frames
    for i in range(5):
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        handler.add_frame(dummy_frame)

        print(f"\nFrame {i+1} added:")
        print(f"  Buffer size: {len(handler)}/{handler.num_frames}")
        print(f"  Ready: {handler.is_ready()}")

        if handler.is_ready():
            tensor = handler.get_temporal_input(device="cpu")
            print(f"  Temporal input shape: {tensor.shape}")
            print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    print("\n✅ TemporalFrameHandler working!")
