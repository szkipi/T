"""Configuration management utilities."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """Load and manage YAML configuration files with variable interpolation."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize config loader.

        Args:
            config_dir: Directory containing config files. Defaults to tennis30/config/
        """
        if config_dir is None:
            # Default to tennis30/config/
            tennis30_root = Path(__file__).parent.parent.parent
            config_dir = tennis30_root / "config"

        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}

    def load(self, config_name: str = "default") -> Dict[str, Any]:
        """Load a configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Interpolate variables
        self._config = self._interpolate_variables(self._config)

        # Override with environment variables
        self._config = self._apply_env_overrides(self._config)

        return self._config

    def _interpolate_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively interpolate ${var.name} references in config.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with interpolated values

        Example:
            >>> config = {
            ...     'paths': {'root': '/data'},
            ...     'file': '${paths.root}/file.txt'
            ... }
            >>> interpolated = self._interpolate_variables(config)
            >>> print(interpolated['file'])
            '/data/file.txt'
        """
        # Convert to string for regex processing
        config_str = yaml.dump(config)

        # Find all ${var.name} patterns
        pattern = r"\$\{([^}]+)\}"
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            matches = re.findall(pattern, config_str)
            if not matches:
                break

            for var_path in matches:
                # Resolve variable path (e.g., "paths.root")
                value = self._resolve_path(config, var_path)
                if value is not None:
                    # Replace ${var.name} with actual value
                    config_str = config_str.replace(f"${{{var_path}}}", str(value))

        # Parse back to dict
        return yaml.safe_load(config_str)

    def _resolve_path(self, config: Dict[str, Any], path: str) -> Any:
        """Resolve a dot-separated path in config dict.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "paths.models_dir")

        Returns:
            Value at path, or None if not found
        """
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override config values with environment variables.

        Environment variables should be prefixed with TENNIS30_ and use
        double underscores for nesting.

        Example:
            TENNIS30_DEVICE=cpu overrides config['environment']['device']
            TENNIS30_BALL_TRACKING__YOLO_V1__WEIGHT=0.3 overrides
                config['ball_tracking']['yolo_v1']['weight']

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        prefix = "TENNIS30_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove prefix and convert to config path
            config_path = env_key[len(prefix) :].lower()
            keys = config_path.split("__")

            # Navigate to the right location
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value (try to parse as int/float/bool)
            final_key = keys[-1]
            current[final_key] = self._parse_env_value(env_value)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value (int, float, bool, or str)
        """
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., "ball_tracking.ensemble.tracknet.weight")
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            >>> loader = ConfigLoader()
            >>> loader.load('default')
            >>> weight = loader.get('ball_tracking.ensemble.tracknet.weight')
            >>> print(weight)  # 0.30
        """
        value = self._resolve_path(self._config, path)
        return value if value is not None else default

    def set(self, path: str, value: Any) -> None:
        """Set a configuration value by dot-separated path.

        Args:
            path: Dot-separated path
            value: Value to set

        Example:
            >>> loader = ConfigLoader()
            >>> loader.load('default')
            >>> loader.set('environment.device', 'cpu')
        """
        keys = path.split(".")
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file.

        Args:
            output_path: Path to save config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config.

        Args:
            key: Configuration key

        Returns:
            Configuration value
        """
        return self._config[key]

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_dir={self.config_dir}, loaded={bool(self._config)})"


# Convenience function
def load_config(config_name: str = "default", config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a configuration file (convenience function).

    Args:
        config_name: Name of config file
        config_dir: Directory containing config files

    Returns:
        Configuration dictionary

    Example:
        >>> from tennis30.utils.config import load_config
        >>> config = load_config('default')
        >>> print(config['environment']['device'])
        'cuda'
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)
