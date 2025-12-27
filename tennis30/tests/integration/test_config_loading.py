"""Integration tests for configuration loading.

Tests that configuration is properly structured and can be loaded
by actual components.
"""

import sys
from pathlib import Path

import pytest

# Add tennis30 to path for testing
tennis30_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tennis30_root))

from src.utils.config import ConfigLoader


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        loader = ConfigLoader()
        config = loader.load("default")

        assert config is not None
        assert "project" in config
        assert "ball_tracking" in config

    def test_ball_tracking_config_structure(self):
        """Test that ball_tracking config has proper structure."""
        loader = ConfigLoader()
        config = loader.load("default")

        ball_config = config.get("ball_tracking", {})

        # Should have ensemble section
        assert "ensemble" in ball_config

        ensemble = ball_config["ensemble"]

        # Should have spatial_clustering
        assert "spatial_clustering" in ensemble
        spatial = ensemble["spatial_clustering"]
        assert "radius" in spatial
        assert "min_models" in spatial

        # Should have individual tracker configs
        assert "tracknet" in ensemble
        assert "yolo_v1" in ensemble
        assert "yolo_v2" in ensemble

    def test_ensemble_tracker_config_keys(self):
        """Test that ensemble tracker configs have required keys."""
        loader = ConfigLoader()
        config = loader.load("default")

        ensemble = config["ball_tracking"]["ensemble"]

        for tracker_name in ["tracknet", "yolo_v1", "yolo_v2"]:
            tracker_config = ensemble[tracker_name]

            assert isinstance(tracker_config, dict), f"{tracker_name} should be dict"
            assert "enabled" in tracker_config
            assert "weight" in tracker_config
            assert "threshold" in tracker_config

    def test_weights_sum_to_one(self):
        """Test that model weights sum to approximately 1.0."""
        loader = ConfigLoader()
        config = loader.load("default")

        ensemble = config["ball_tracking"]["ensemble"]

        total_weight = 0.0
        for key, value in ensemble.items():
            if isinstance(value, dict) and "weight" in value:
                if value.get("enabled", True):
                    total_weight += value["weight"]

        # Should sum to 1.0 (within floating point error)
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected ~1.0"

    def test_spatial_clustering_values(self):
        """Test that spatial clustering parameters are reasonable."""
        loader = ConfigLoader()
        config = loader.load("default")

        spatial = config["ball_tracking"]["ensemble"]["spatial_clustering"]

        radius = spatial["radius"]
        min_models = spatial["min_models"]

        assert radius > 0, "Radius must be positive"
        assert radius < 100, "Radius seems too large"
        assert min_models >= 1, "min_models must be at least 1"
        assert min_models <= 5, "min_models can't exceed number of models"

    def test_config_get_with_path(self):
        """Test getting nested config values."""
        loader = ConfigLoader()
        loader.load("default")

        # Test dot-separated path access
        tracknet_weight = loader.get("ball_tracking.ensemble.tracknet.weight")
        assert tracknet_weight == 0.30

        radius = loader.get("ball_tracking.ensemble.spatial_clustering.radius")
        assert radius == 10

    def test_physics_config_structure(self):
        """Test physics configuration structure."""
        loader = ConfigLoader()
        config = loader.load("default")

        physics = config.get("physics", {})

        assert "gravity" in physics
        assert physics["gravity"] == 9.81

        assert "drag_coefficient" in physics
        assert "ball_properties" in physics

        ball_props = physics["ball_properties"]
        assert "mass" in ball_props
        assert "radius" in ball_props


class TestConfigUsability:
    """Test that configuration works with actual components."""

    def test_config_for_ensemble_tracker(self):
        """Test config structure matches EnsembleBallTracker expectations.

        This tests the fix for BUG #1.
        """
        loader = ConfigLoader()
        config = loader.load("default")

        ball_config = config.get("ball_tracking", {})

        # This is what EnsembleBallTracker receives
        ensemble_config = ball_config.get("ensemble", {})

        # Should be able to access spatial_clustering from ensemble_config
        spatial_config = ensemble_config.get("spatial_clustering", {})

        assert spatial_config.get("radius") == 10
        assert spatial_config.get("min_models") == 2

        # Should be able to iterate over tracker configs
        tracker_configs = {
            name: cfg
            for name, cfg in ensemble_config.items()
            if isinstance(cfg, dict) and name != "spatial_clustering"
        }

        assert len(tracker_configs) >= 3  # At least tracknet, yolo_v1, yolo_v2
        assert "tracknet" in tracker_configs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
