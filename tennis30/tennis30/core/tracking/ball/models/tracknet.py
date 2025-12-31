"""TrackNet VGG-16 encoder-decoder architecture for ball tracking.

TrackNet uses a VGG-16 based encoder-decoder to generate heatmaps
for tennis ball detection. It processes 3 consecutive frames to
capture temporal information.

Reference:
    - Paper: "TrackNet: A Deep Learning Network for Tracking High-speed
             and Tiny Objects in Sports Applications"
    - Original: Keras/TensorFlow implementation
    - This: PyTorch re-implementation
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNetVGG(nn.Module):
    """TrackNet VGG-16 encoder-decoder for ball heatmap generation.

    Architecture:
        - Input: 3 frames (t-1, t, t+1) concatenated → 9 channels
        - Encoder: VGG-16 convolutional layers
        - Decoder: Deconvolution layers with skip connections
        - Output: Single-channel heatmap (512x512)

    The heatmap has a Gaussian blob centered on the ball position.

    Attributes:
        input_height: Input image height (default: 512)
        input_width: Input image width (default: 512)
        num_frames: Number of frames to process (default: 3)
    """

    def __init__(
        self,
        input_height: int = 512,
        input_width: int = 512,
        num_frames: int = 3,
    ):
        """Initialize TrackNet model.

        Args:
            input_height: Input image height
            input_width: Input image width
            num_frames: Number of consecutive frames (3 for temporal context)
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.num_frames = num_frames
        self.input_channels = num_frames * 3  # RGB * num_frames

        # =================================================================
        # ENCODER (VGG-16 based)
        # =================================================================

        # Block 1
        self.conv1_1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # =================================================================
        # DECODER (Deconvolution with skip connections)
        # =================================================================

        # Unpool 4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.deconv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.deconv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        # Unpool 3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Unpool 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.deconv2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Unpool 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.deconv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

        # Activation
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TrackNet.

        Args:
            x: Input tensor (B, C, H, W) where C = num_frames * 3
               Shape: (batch_size, 9, 512, 512) for 3 RGB frames

        Returns:
            Heatmap tensor (B, 1, H, W)
            Shape: (batch_size, 1, 512, 512)
        """
        # =================================================================
        # ENCODER
        # =================================================================

        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x, indices1 = self.pool1(x)

        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x, indices2 = self.pool2(x)

        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x, indices3 = self.pool3(x)

        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x, indices4 = self.pool4(x)

        # Block 5 (no pooling)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))

        # =================================================================
        # DECODER
        # =================================================================

        # Unpool 4
        x = self.unpool4(x, indices4)
        x = self.relu(self.deconv4_1(x))
        x = self.relu(self.deconv4_2(x))
        x = self.relu(self.deconv4_3(x))

        # Unpool 3
        x = self.unpool3(x, indices3)
        x = self.relu(self.deconv3_1(x))
        x = self.relu(self.deconv3_2(x))
        x = self.relu(self.deconv3_3(x))

        # Unpool 2
        x = self.unpool2(x, indices2)
        x = self.relu(self.deconv2_1(x))
        x = self.relu(self.deconv2_2(x))

        # Unpool 1
        x = self.unpool1(x, indices1)
        x = self.relu(self.deconv1_1(x))
        x = self.relu(self.deconv1_2(x))

        # Output
        x = self.output(x)
        x = self.sigmoid(x)

        return x

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Get model summary."""
        total_params = self.get_num_params()
        return (
            f"TrackNetVGG(\n"
            f"  Input: {self.num_frames} frames × 3 channels = {self.input_channels} channels\n"
            f"  Resolution: {self.input_height}×{self.input_width}\n"
            f"  Output: 1-channel heatmap\n"
            f"  Parameters: {total_params:,}\n"
            f")"
        )


def create_tracknet(
    input_size: Tuple[int, int] = (512, 512),
    num_frames: int = 3,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
) -> TrackNetVGG:
    """Create TrackNet model.

    Args:
        input_size: (height, width) of input images
        num_frames: Number of consecutive frames
        pretrained: Whether to load pretrained weights
        weights_path: Path to .pth weights file

    Returns:
        TrackNetVGG model instance
    """
    model = TrackNetVGG(
        input_height=input_size[0],
        input_width=input_size[1],
        num_frames=num_frames,
    )

    if pretrained and weights_path:
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"✓ Loaded TrackNet weights from: {weights_path}")
        except Exception as e:
            print(f"⚠ Failed to load weights: {e}")
            print(f"  Continuing with random initialization")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_tracknet()
    print(model.summary())

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 9, 512, 512)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
