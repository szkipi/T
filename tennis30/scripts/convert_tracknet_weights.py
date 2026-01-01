#!/usr/bin/env python3
"""Convert TrackNet weights from Keras to PyTorch.

This script converts the original TrackNet Keras model weights
to PyTorch state_dict format for use with the PyTorch implementation.

Usage:
    python scripts/convert_tracknet_weights.py \
        --keras-model path/to/model.1 \
        --output path/to/tracknet.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def load_keras_weights(keras_path: str) -> Dict[str, np.ndarray]:
    """Load weights from Keras model file.

    Args:
        keras_path: Path to Keras .h5 or model.1 file

    Returns:
        Dictionary mapping layer names to weight arrays
    """
    try:
        import h5py
    except ImportError:
        print("❌ h5py not installed. Install with: pip install h5py")
        sys.exit(1)

    weights = {}

    with h5py.File(keras_path, 'r') as f:
        print(f"📂 Loading Keras model from: {keras_path}")
        print(f"   Keys: {list(f.keys())}")

        # Keras model structure: model_weights/layer_name/layer_name/weights
        if 'model_weights' in f:
            model_weights = f['model_weights']

            for layer_name in model_weights.keys():
                layer_group = model_weights[layer_name]

                if layer_name in layer_group:
                    layer_params = layer_group[layer_name]

                    # Get weights and biases
                    if 'kernel:0' in layer_params:
                        weights[f"{layer_name}.weight"] = layer_params['kernel:0'][:]
                    if 'bias:0' in layer_params:
                        weights[f"{layer_name}.bias"] = layer_params['bias:0'][:]

        # Alternative structure
        elif 'layer_names' in f.attrs:
            layer_names = f.attrs['layer_names']
            for layer_name in layer_names:
                layer_name_str = layer_name.decode('utf-8')
                layer_group = f[layer_name_str]

                # Extract weight tensors
                if hasattr(layer_group, 'keys'):
                    for key in layer_group.keys():
                        data = layer_group[key][:]
                        weights[f"{layer_name_str}.{key}"] = data

    print(f"✓ Loaded {len(weights)} weight tensors")
    return weights


def map_keras_to_pytorch(keras_weights: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Map Keras weights to PyTorch state_dict.

    Keras Conv2D weights shape: (height, width, in_channels, out_channels)
    PyTorch Conv2d weights shape: (out_channels, in_channels, height, width)

    Args:
        keras_weights: Dictionary of Keras weights

    Returns:
        PyTorch state_dict
    """
    pytorch_state = {}

    # Mapping of Keras layer names to PyTorch module names
    layer_mapping = {
        # Encoder
        'conv1_1': 'conv1_1',
        'conv1_2': 'conv1_2',
        'conv2_1': 'conv2_1',
        'conv2_2': 'conv2_2',
        'conv3_1': 'conv3_1',
        'conv3_2': 'conv3_2',
        'conv3_3': 'conv3_3',
        'conv4_1': 'conv4_1',
        'conv4_2': 'conv4_2',
        'conv4_3': 'conv4_3',
        'conv5_1': 'conv5_1',
        'conv5_2': 'conv5_2',
        'conv5_3': 'conv5_3',
        # Decoder
        'deconv4_1': 'deconv4_1',
        'deconv4_2': 'deconv4_2',
        'deconv4_3': 'deconv4_3',
        'deconv3_1': 'deconv3_1',
        'deconv3_2': 'deconv3_2',
        'deconv3_3': 'deconv3_3',
        'deconv2_1': 'deconv2_1',
        'deconv2_2': 'deconv2_2',
        'deconv1_1': 'deconv1_1',
        'deconv1_2': 'deconv1_2',
        # Output
        'output': 'output',
    }

    print("\n🔄 Converting weights to PyTorch format...")

    for keras_name, pytorch_name in layer_mapping.items():
        # Look for weight
        weight_key = f"{keras_name}.weight"
        bias_key = f"{keras_name}.bias"

        # Try different naming conventions
        keras_weight_keys = [
            weight_key,
            f"{keras_name}.kernel",
            f"{keras_name}/kernel:0",
        ]

        keras_bias_keys = [
            bias_key,
            f"{keras_name}.bias",
            f"{keras_name}/bias:0",
        ]

        # Find weight
        weight_data = None
        for key in keras_weight_keys:
            if key in keras_weights:
                weight_data = keras_weights[key]
                break

        if weight_data is not None:
            # Convert shape: (H, W, C_in, C_out) → (C_out, C_in, H, W)
            if len(weight_data.shape) == 4:
                weight_data = np.transpose(weight_data, (3, 2, 0, 1))

            pytorch_state[f"{pytorch_name}.weight"] = torch.from_numpy(weight_data.copy())
            print(f"  ✓ {pytorch_name}.weight: {pytorch_state[f'{pytorch_name}.weight'].shape}")

        # Find bias
        bias_data = None
        for key in keras_bias_keys:
            if key in keras_weights:
                bias_data = keras_weights[key]
                break

        if bias_data is not None:
            pytorch_state[f"{pytorch_name}.bias"] = torch.from_numpy(bias_data.copy())
            print(f"  ✓ {pytorch_name}.bias: {pytorch_state[f'{pytorch_name}.bias'].shape}")

    print(f"\n✓ Converted {len(pytorch_state)} tensors to PyTorch format")
    return pytorch_state


def verify_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Verify converted weights are valid.

    Args:
        state_dict: PyTorch state_dict

    Returns:
        True if weights are valid
    """
    print("\n🔍 Verifying weights...")

    # Check for NaN or Inf
    for name, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            print(f"  ❌ {name}: Contains NaN values")
            return False
        if torch.isinf(tensor).any():
            print(f"  ❌ {name}: Contains Inf values")
            return False

    print("  ✓ No NaN or Inf values found")

    # Check weight shapes match expected
    expected_shapes = {
        'conv1_1.weight': (64, 9, 3, 3),  # 9 input channels (3 frames × 3 RGB)
        'output.weight': (1, 64, 1, 1),   # 1 output channel (heatmap)
    }

    for name, expected_shape in expected_shapes.items():
        if name in state_dict:
            actual_shape = tuple(state_dict[name].shape)
            if actual_shape != expected_shape:
                print(f"  ❌ {name}: Expected {expected_shape}, got {actual_shape}")
                return False
            print(f"  ✓ {name}: Shape {actual_shape} correct")

    print("✓ Weights verified successfully")
    return True


def test_converted_model(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Test the converted weights with a forward pass.

    Args:
        state_dict: PyTorch state_dict

    Returns:
        True if forward pass succeeds
    """
    print("\n🧪 Testing converted model...")

    try:
        from tennis30.core.tracking.ball.models.tracknet import create_tracknet

        # Create model
        model = create_tracknet(
            input_size=(512, 512),
            num_frames=3,
            pretrained=False,
        )

        # Load converted weights
        model.load_state_dict(state_dict)
        model.eval()

        # Test forward pass
        dummy_input = torch.randn(1, 9, 512, 512)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"  ✓ Forward pass successful")
        print(f"    Input shape:  {dummy_input.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Check output is valid heatmap
        if output.shape != (1, 1, 512, 512):
            print(f"  ❌ Output shape incorrect")
            return False

        if output.min() < 0 or output.max() > 1:
            print(f"  ⚠️  Output range outside [0, 1] (sigmoid issue?)")

        print("✓ Model test passed")
        return True

    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert TrackNet Keras weights to PyTorch"
    )
    parser.add_argument(
        "--keras-model",
        type=str,
        required=True,
        help="Path to Keras model file (.h5 or model.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tennis30/data/models/tracknet.pth",
        help="Output path for PyTorch weights (.pth)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip weight verification",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip model testing",
    )

    args = parser.parse_args()

    # Check input exists
    keras_path = Path(args.keras_model)
    if not keras_path.exists():
        print(f"❌ Keras model not found: {keras_path}")
        sys.exit(1)

    # Load Keras weights
    keras_weights = load_keras_weights(str(keras_path))

    if not keras_weights:
        print("❌ No weights loaded from Keras model")
        sys.exit(1)

    # Convert to PyTorch
    pytorch_state = map_keras_to_pytorch(keras_weights)

    if not pytorch_state:
        print("❌ Conversion failed - no weights converted")
        sys.exit(1)

    # Verify weights
    if not args.no_verify:
        if not verify_weights(pytorch_state):
            print("❌ Weight verification failed")
            sys.exit(1)

    # Test model
    if not args.no_test:
        if not test_converted_model(pytorch_state):
            print("⚠️  Model test failed - weights may not be compatible")
            print("   Saving anyway, but model may not work correctly")

    # Save PyTorch weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(pytorch_state, output_path)
    print(f"\n✅ Weights saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Print usage instructions
    print("\n📝 Usage:")
    print(f"   Update your config to use: model_path: '{output_path}'")
    print(f"   Or pass to create_tracknet: weights_path='{output_path}'")


if __name__ == "__main__":
    main()
