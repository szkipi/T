"""Unreal Engine Export Format"""

import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class UnrealExporter:
    """Export to Unreal Engine compatible format."""

    def export(self, results: Dict, output_path: str):
        """Export results for Unreal Engine."""

        unreal_data = {
            'Version': '1.0',
            'Court': results['court'],
            'Ball': results['ball'],
            'Players': results['players'],
            'GameState': results['game_state']
        }

        with open(output_path, 'w') as f:
            json.dump(unreal_data, f, indent=2, default=str)

        logger.info(f"Unreal export saved to {output_path}")
