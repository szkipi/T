#!/usr/bin/env python3
"""Download pretrained model weights for Tennis30.

This script downloads all required model weights from their respective sources
and places them in the data/models/ directory.

Usage:
    # Download all models
    python scripts/download_models.py --all

    # Download specific models
    python scripts/download_models.py tracknet yolov8n_tennis_v1

    # Download only ball tracking models
    python scripts/download_models.py --type ball_tracker

    # Force re-download even if file exists
    python scripts/download_models.py --all --force
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

# Add tennis30 to path
tennis30_root = Path(__file__).parent.parent
sys.path.insert(0, str(tennis30_root))

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please run: pip install requests tqdm")
    sys.exit(1)

from src.models.registry import MODEL_METADATA, list_available_models


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hexadecimal checksum string
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url: str, output_path: Path, expected_checksum: Optional[str] = None) -> bool:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save file
        expected_checksum: Optional SHA256 checksum to verify

    Returns:
        True if download successful, False otherwise
    """
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {output_path.name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify checksum if provided
        if expected_checksum:
            print(f"Verifying checksum for {output_path.name}...", end=" ")
            actual_checksum = compute_checksum(output_path)
            if actual_checksum != expected_checksum:
                print("FAILED")
                print(f"  Expected: {expected_checksum}")
                print(f"  Got:      {actual_checksum}")
                output_path.unlink()  # Delete corrupted file
                return False
            print("OK")

        return True

    except requests.RequestException as e:
        print(f"ERROR downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_model(
    model_name: str, models_dir: Path, force: bool = False
) -> bool:
    """Download a single model.

    Args:
        model_name: Name of model to download
        models_dir: Directory to save models
        force: Force re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    metadata = MODEL_METADATA.get(model_name)
    if not metadata:
        print(f"ERROR: Unknown model '{model_name}'")
        return False

    filename = metadata["filename"]
    output_path = models_dir / filename

    # Check if already exists
    if output_path.exists() and not force:
        print(f"✓ {model_name} already exists at {output_path}")
        return True

    # Check if URL is available
    url = metadata.get("url")
    if not url:
        print(f"⚠ {model_name}: No download URL (manual installation required)")
        return False

    print(f"⬇️  Downloading {model_name} ({metadata['size_mb']} MB)...")
    print(f"   Description: {metadata['description']}")
    print(f"   URL: {url}")

    success = download_file(url, output_path, metadata.get("checksum"))

    if success:
        print(f"✓ {model_name} downloaded successfully to {output_path}")
        print()
    else:
        print(f"✗ Failed to download {model_name}")
        print()

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Tennis30 model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "models",
        nargs="*",
        help="Specific models to download (if not using --all or --type)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--type",
        choices=["ball_tracker", "court_detector", "pose_estimator", "physics", "player_tracker"],
        help="Download all models of a specific type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to save models (default: tennis30/data/models/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    # Determine models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = tennis30_root / "data" / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    # List models if requested
    if args.list:
        print("Available models:")
        print()
        for model_type in ["ball_tracker", "player_tracker", "pose_estimator", "physics"]:
            models = list_available_models(model_type)
            if models:
                print(f"  {model_type.upper()}:")
                for name, meta in models.items():
                    url_status = "✓" if meta.get("url") else "⚠ (manual)"
                    print(f"    - {name:25} {meta['size_mb']:6} MB  {url_status}")
                    print(f"      {meta['description']}")
                print()
        return

    # Determine which models to download
    models_to_download: List[str] = []

    if args.all:
        models_to_download = list(MODEL_METADATA.keys())
    elif args.type:
        type_models = list_available_models(args.type)
        models_to_download = list(type_models.keys())
    elif args.models:
        models_to_download = args.models
    else:
        parser.print_help()
        return

    if not models_to_download:
        print("No models to download.")
        return

    # Download models
    print(f"Models directory: {models_dir}")
    print(f"Downloading {len(models_to_download)} model(s)...")
    print()

    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, models_dir, force=args.force):
            success_count += 1

    # Summary
    print("=" * 60)
    print(f"Downloaded {success_count}/{len(models_to_download)} models successfully")

    if success_count < len(models_to_download):
        print()
        print("Some models failed to download. Please check the errors above.")
        print("You may need to download some models manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
