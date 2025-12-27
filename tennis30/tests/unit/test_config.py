"""Unit tests for configuration management."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from tennis30.src.utils.config import ConfigLoader, load_config


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        loader = ConfigLoader()
        config = loader.load("default")

        assert config is not None
        assert "project" in config
        assert "environment" in config
        assert "ball_tracking" in config

    def test_get_value_by_path(self):
        """Test getting config values by dot-separated path."""
        loader = ConfigLoader()
        loader.load("default")

        # Get nested value
        weight = loader.get("ball_tracking.ensemble.tracknet.weight")
        assert weight == 0.30

        # Get top-level value
        device = loader.get("environment.device")
        assert device in ["cuda", "cpu"]

    def test_get_nonexistent_key_returns_default(self):
        """Test that nonexistent keys return the default value."""
        loader = ConfigLoader()
        loader.load("default")

        value = loader.get("nonexistent.key", default="DEFAULT")
        assert value == "DEFAULT"

    def test_set_value_by_path(self):
        """Test setting config values by dot-separated path."""
        loader = ConfigLoader()
        loader.load("default")

        # Set a value
        loader.set("environment.device", "cpu")
        assert loader.get("environment.device") == "cpu"

        # Set a nested value
        loader.set("ball_tracking.ensemble.tracknet.weight", 0.40)
        assert loader.get("ball_tracking.ensemble.tracknet.weight") == 0.40

    def test_variable_interpolation(self):
        """Test that ${var.name} variables are interpolated."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "test.yaml"

            # Create config with variable references
            config_data = {
                "paths": {"root": "/data"},
                "models_dir": "${paths.root}/models",
                "checkpoint": "${models_dir}/model.pth",
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Load and check interpolation
            loader = ConfigLoader(config_dir)
            config = loader.load("test")

            assert config["models_dir"] == "/data/models"
            assert config["checkpoint"] == "/data/models/model.pth"

    def test_env_override(self):
        """Test that environment variables override config values."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "test.yaml"

            config_data = {"environment": {"device": "cuda"}}

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Set environment variable
            os.environ["TENNIS30_ENVIRONMENT__DEVICE"] = "cpu"

            try:
                loader = ConfigLoader(config_dir)
                config = loader.load("test")

                assert config["environment"]["device"] == "cpu"
            finally:
                # Clean up
                del os.environ["TENNIS30_ENVIRONMENT__DEVICE"]

    def test_env_value_parsing(self):
        """Test that environment values are parsed correctly."""
        loader = ConfigLoader()

        # Boolean
        assert loader._parse_env_value("true") is True
        assert loader._parse_env_value("false") is False
        assert loader._parse_env_value("yes") is True
        assert loader._parse_env_value("no") is False

        # Integer
        assert loader._parse_env_value("42") == 42

        # Float
        assert loader._parse_env_value("3.14") == 3.14

        # String
        assert loader._parse_env_value("hello") == "hello"

    def test_save_config(self):
        """Test saving configuration to file."""
        with TemporaryDirectory() as tmpdir:
            loader = ConfigLoader()
            loader.load("default")

            # Modify a value
            loader.set("environment.device", "cpu")

            # Save to file
            output_file = Path(tmpdir) / "saved_config.yaml"
            loader.save(output_file)

            # Load saved file and verify
            assert output_file.exists()
            with open(output_file) as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["environment"]["device"] == "cpu"

    def test_dict_access(self):
        """Test dictionary-style access to config."""
        loader = ConfigLoader()
        loader.load("default")

        # Access like a dictionary
        assert "project" in loader["project"]
        assert loader["environment"]["device"] in ["cuda", "cpu"]

    def test_config_property(self):
        """Test the config property getter."""
        loader = ConfigLoader()
        loader.load("default")

        config = loader.config
        assert isinstance(config, dict)
        assert "project" in config


class TestLoadConfigFunction:
    """Tests for load_config convenience function."""

    def test_load_config_default(self):
        """Test loading default config with convenience function."""
        config = load_config("default")

        assert config is not None
        assert "project" in config
        assert "environment" in config

    def test_load_config_nonexistent_file(self):
        """Test that loading nonexistent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")


class TestConfigIntegration:
    """Integration tests for full config workflow."""

    def test_full_workflow(self):
        """Test complete config workflow: load, modify, save, reload."""
        with TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create initial config
            initial_config = {
                "test_value": 42,
                "nested": {"value": "hello"},
            }

            config_file = config_dir / "workflow.yaml"
            with open(config_file, "w") as f:
                yaml.dump(initial_config, f)

            # Load
            loader = ConfigLoader(config_dir)
            config = loader.load("workflow")
            assert config["test_value"] == 42

            # Modify
            loader.set("test_value", 100)
            loader.set("nested.value", "world")

            # Save
            output_file = config_dir / "modified.yaml"
            loader.save(output_file)

            # Reload and verify
            loader2 = ConfigLoader(config_dir)
            config2 = loader2.load("modified")

            assert config2["test_value"] == 100
            assert config2["nested"]["value"] == "world"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
