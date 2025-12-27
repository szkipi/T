#!/usr/bin/env python3
"""
Tennis30 - Quick Test Script for Uploaded Files
================================================
Használat: python test_upload.py <file_path>

Támogatott formátumok:
- Képek: .jpg, .png, .jpeg
- Videók: .mp4, .avi, .mov
"""

import sys
import os
sys.path.insert(0, '/home/user/T/tennis30/src')

import cv2
import numpy as np
from pathlib import Path


def test_single_frame(image_path: str):
    """Test ball detection on a single frame"""
    print(f"📸 Testing single frame: {image_path}\n")

    # Check if dependencies are available
    try:
        from core.tracking.ball.ensemble import EnsembleBallTracker
        from utils.config import ConfigLoader

        # Load config
        config = ConfigLoader().load("default")
        ball_config = config.get("ball_tracking", {})

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            return

        print(f"✓ Image loaded: {frame.shape[1]}x{frame.shape[0]}")

        # Initialize tracker (this will show which models are available)
        print("\n🔧 Initializing ensemble tracker...")
        tracker = EnsembleBallTracker(ball_config, device="cpu")

        # Detect ball
        print("\n🎾 Detecting ball...\n")
        detection = tracker.detect(frame)

        # Display results
        print("=" * 60)
        print("DETECTION RESULTS:")
        print("=" * 60)
        print(f"Position: {detection.get('position', 'N/A')}")
        print(f"Confidence: {detection.get('confidence', 0):.2%}")
        print(f"Bounding box: {detection.get('bbox', 'N/A')}")
        print(f"Models voted: {detection.get('num_models', 0)}")
        print("=" * 60)

        # Visualize if ball detected
        if detection.get('position') is not None:
            visualize_detection(frame, detection, image_path)

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n💡 Install dependencies first:")
        print("   pip install numpy opencv-python pyyaml")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_video(video_path: str, max_frames: int = 30):
    """Test ball tracking on video (first N frames)"""
    print(f"🎬 Testing video: {video_path}\n")

    try:
        from core.tracking.ball.ensemble import EnsembleBallTracker
        from utils.config import ConfigLoader

        # Load config
        config = ConfigLoader().load("default")
        ball_config = config.get("ball_tracking", {})

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✓ Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        print(f"  Processing first {min(max_frames, total_frames)} frames...\n")

        # Initialize tracker
        print("🔧 Initializing ensemble tracker...")
        tracker = EnsembleBallTracker(ball_config, device="cpu")

        # Process frames
        print("\n🎾 Tracking ball...\n")
        detections = []
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            detection = tracker.detect(frame)
            detections.append(detection)

            if frame_idx % 10 == 0:
                conf = detection.get('confidence', 0)
                pos = detection.get('position', None)
                status = "✓" if pos is not None else "✗"
                print(f"  Frame {frame_idx:3d}: {status} conf={conf:.2%} pos={pos}")

            frame_idx += 1

        cap.release()

        # Summary
        successful = sum(1 for d in detections if d.get('position') is not None)
        print("\n" + "=" * 60)
        print("TRACKING SUMMARY:")
        print("=" * 60)
        print(f"Frames processed: {frame_idx}")
        print(f"Ball detected: {successful} / {frame_idx} ({successful/frame_idx*100:.1f}%)")
        print(f"Average confidence: {np.mean([d.get('confidence', 0) for d in detections]):.2%}")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n💡 Install dependencies first:")
        print("   pip install numpy opencv-python pyyaml")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def visualize_detection(frame, detection, original_path):
    """Draw detection on frame and save"""
    output_path = str(Path(original_path).with_name(
        Path(original_path).stem + "_detection.jpg"
    ))

    vis_frame = frame.copy()
    pos = detection.get('position')
    bbox = detection.get('bbox')
    conf = detection.get('confidence', 0)

    if pos is not None:
        # Draw ball position
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(vis_frame, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)

        # Draw bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw confidence text
        cv2.putText(
            vis_frame,
            f"Ball: {conf:.1%}",
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, vis_frame)
    print(f"\n💾 Visualization saved: {output_path}")


def simple_color_based_test(image_path: str):
    """Fallback: Simple color-based ball detection (no models needed)"""
    print(f"🎾 Simple color-based test: {image_path}\n")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    print(f"✓ Image loaded: {frame.shape[1]}x{frame.shape[0]}")

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tennis ball yellow-green range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n🔍 Found {len(contours)} yellow-green objects")

    if contours:
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 20:  # Minimum area threshold
            # Get center
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                print(f"\n🎾 Potential ball detected!")
                print(f"   Position: ({cx}, {cy})")
                print(f"   Area: {area:.0f} pixels")

                # Visualize
                vis_frame = frame.copy()
                cv2.circle(vis_frame, (cx, cy), 15, (0, 255, 0), 2)
                cv2.drawContours(vis_frame, [largest], -1, (255, 0, 0), 2)

                output_path = str(Path(image_path).with_name(
                    Path(image_path).stem + "_simple_detection.jpg"
                ))
                cv2.imwrite(output_path, vis_frame)
                print(f"\n💾 Visualization saved: {output_path}")
    else:
        print("\n❌ No tennis ball detected (simple color method)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <file_path>")
        print("\nExample:")
        print("  python test_upload.py tennis_frame.jpg")
        print("  python test_upload.py tennis_video.mp4")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    ext = Path(file_path).suffix.lower()

    print("=" * 60)
    print("TENNIS30 - QUICK TEST")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Type: {ext}")
    print("=" * 60)
    print()

    # Check what's available
    print("🔍 Checking dependencies...")
    deps_ok = True
    try:
        import numpy
        print("  ✓ NumPy")
    except:
        print("  ✗ NumPy - install: pip install numpy")
        deps_ok = False

    try:
        import cv2
        print("  ✓ OpenCV")
    except:
        print("  ✗ OpenCV - install: pip install opencv-python")
        deps_ok = False

    try:
        import yaml
        print("  ✓ PyYAML")
    except:
        print("  ✗ PyYAML - install: pip install pyyaml")
        deps_ok = False

    print()

    if not deps_ok:
        print("⚠️  Missing dependencies. Running simple color-based test only.\n")
        if ext in ['.jpg', '.jpeg', '.png']:
            simple_color_based_test(file_path)
        return

    # Run appropriate test
    if ext in ['.jpg', '.jpeg', '.png']:
        test_single_frame(file_path)
    elif ext in ['.mp4', '.avi', '.mov']:
        test_video(file_path, max_frames=30)
    else:
        print(f"❌ Unsupported file format: {ext}")
        print("   Supported: .jpg, .png, .mp4, .avi, .mov")


if __name__ == "__main__":
    main()
