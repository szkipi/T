"""Unity Export Format"""

import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class UnityExporter:
    """Export to Unity-compatible JSON format."""

    def export(self, results: Dict, output_path: str):
        """Export results for Unity game engine."""

        unity_data = {
            'metadata': results['metadata'],
            'court': {
                'dimensions': results['court'].get('dimensions', {}),
                'keypoints': results['court'].get('keypoints_2d', [])
            },
            'ball_trajectory': self._format_ball_trajectory(results['ball']),
            'players': self._format_players(results['players']),
            'game_state': results['game_state']
        }

        with open(output_path, 'w') as f:
            json.dump(unity_data, f, indent=2, default=str)

        logger.info(f"Unity export saved to {output_path}")

    def _format_ball_trajectory(self, ball_data):
        """Format ball data for Unity."""
        # TODO: Convert to Unity coordinate system
        return ball_data

    def _format_players(self, player_data):
        """Format player skeleton for Unity."""
        # TODO: Convert keypoints to Unity format
        return player_data
