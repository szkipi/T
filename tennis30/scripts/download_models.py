#!/usr/bin/env python3
"""Download pre-trained model weights for Tennis30.

Downloads TrackNet and YOLOv8 model weights from various sources.

Usage:
    python scripts/download_models.py tracknet
    python scripts/download_models.py yolo
    python scripts/download_models.py all
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save file
        description: Description for progress display

    Returns:
        True if successful
    """
    try:
        print(f"\n📥 Downloading {description}...")
        print(f"   From: {url}")
        print(f"   To:   {output_path}")

        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                size_mb = total_size / 1024 / 1024
                downloaded_mb = downloaded / 1024 / 1024
                print(f"   Progress: {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", end='\r')

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_tracknet_keras(output_dir: Path) -> bool:
    """Download TrackNet Keras model weights.

    Args:
        output_dir: Directory to save model

    Returns:
        True if successful
    """
    print("\n🎾 TrackNet Keras Weights")
    print("=" * 60)

    # Known sources for TrackNet weights
    output_path = output_dir / "tracknet_keras.h5"

    if output_path.exists():
        print(f"\n✓ Model already exists: {output_path}")
        return True

    print("\n⚠️  TrackNet Keras weights require manual download")
    print("   Please follow these steps:")
    print()
    print("   OPTION 1: Clone tennis-tracking repo")
    print("   $ git clone https://github.com/yastrebovd/tennis-tracking")
    print("   $ cp tennis-tracking/TrackNet/model.1 tennis30/data/models/tracknet_keras.h5")
    print()
    print("   OPTION 2: Direct download from GitHub")
    print("   $ curl -L https://github.com/yastrebovd/tennis-tracking/raw/master/TrackNet/model.1 \\")
    print(f"     -o {output_path}")
    print()
    print("   Then run the conversion script:")
    print(f"   $ python scripts/convert_tracknet_weights.py \\")
    print(f"       --keras-model {output_path} \\")
    print(f"       --output tennis30/data/models/tracknet.pth")

    return False


def download_yolo_weights(output_dir: Path, version: str = "v1") -> bool:
    """Download YOLOv8 tennis ball detection weights.

    Args:
        output_dir: Directory to save model
        version: Model version (v1 or v2)

    Returns:
        True if successful
    """
    print(f"\n🎾 YOLOv8 Tennis Ball Weights ({version})")
    print("=" * 60)

    # YOLOv8 tennis ball weights sources
    sources = {
        "v1": {
            "name": "YOLOv8n Tennis Ball v1",
            "filename": "yolo_tennis_v1.pt",
        },
        "v2": {
            "name": "YOLOv8n Tennis Ball v2",
            "filename": "yolo_tennis_v2.pt",
        },
    }

    if version not in sources:
        print(f"❌ Unknown version: {version}")
        print(f"   Available: {', '.join(sources.keys())}")
        return False

    source = sources[version]
    output_path = output_dir / source["filename"]

    if output_path.exists():
        print(f"\n✓ Model already exists: {output_path}")
        return True

    print("\n⚠️  YOLOv8 weights require manual download or training")
    print("   Please follow these steps:")
    print()
    print("   OPTION 1: Train your own YOLOv8 model")
    print("   $ yolo task=detect mode=train model=yolov8n.pt data=tennis_ball.yaml epochs=100")
    print()
    print("   OPTION 2: Use pre-trained YOLOv8n (general object detection)")
    print("   $ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
    print(f"   $ mv yolov8n.pt {output_path}")
    print()
    print("   Note: General YOLOv8n will work but may not detect small tennis balls well.")
    print("   For best results, train on tennis ball dataset.")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Tennis30 model weights"
    )
    parser.add_argument(
        "model",
        choices=["tracknet", "yolo", "yolo-v1", "yolo-v2", "all"],
        help="Which model weights to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tennis30/data/models",
        help="Output directory for models",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tennis30 Model Downloader")
    print("=" * 60)

    if args.model in ["tracknet", "all"]:
        download_tracknet_keras(output_dir)

    if args.model in ["yolo", "yolo-v1", "all"]:
        download_yolo_weights(output_dir, "v1")

    if args.model in ["yolo", "yolo-v2", "all"]:
        download_yolo_weights(output_dir, "v2")

    print("\n" + "=" * 60)
    print("📝 Manual Download Instructions")
    print("=" * 60)
    print("\nFor TrackNet weights:")
    print("  https://github.com/yastrebovd/tennis-tracking")
    print("\nFor YOLOv8 weights:")
    print("  Train custom model or use general YOLOv8n")
    print("=" * 60)


if __name__ == "__main__":
    main()
