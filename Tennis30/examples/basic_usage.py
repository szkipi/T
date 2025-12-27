"""
Tennis30 Basic Usage Example

Process a tennis match video with precision multi-pass pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Tennis30 import PrecisionPipeline


def main():
    """Example: Process tennis match video."""

    # Initialize pipeline
    print("=" * 60)
    print("Tennis30 - Precision Tennis Tracking")
    print("=" * 60)

    pipeline = PrecisionPipeline(
        config_path='config/precision_config.yaml',
        mode='precision',  # 'fast', 'balanced', or 'precision'
        passes=4,          # 1-4 refinement passes
        device='cuda'      # 'cuda' or 'cpu'
    )

    # Process video
    results = pipeline.process_video(
        video_path='input/match.mp4',
        output_dir='output/match_001',
        start_frame=0,
        end_frame=None,  # None = process entire video
        visualize=True   # Generate visualization video
    )

    # Results automatically exported to:
    # - output/match_001/ball_data.csv
    # - output/match_001/player1_data.csv
    # - output/match_001/player2_data.csv
    # - output/match_001/game_state.json
    # - output/match_001/unity_export.json
    # - output/match_001/unreal_export.json

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Quality Score: {results['metadata']['quality_score']:.2%}")
    print(f"Total Frames: {results['metadata']['total_frames']}")
    print(f"Duration: {results['metadata']['duration']:.1f}s")
    print(f"Rallies: {len(results['game_state']['rallies'])}")
    print(f"Output Directory: output/match_001")
    print("=" * 60)


if __name__ == '__main__':
    main()
