"""
Configuration Loader for ProSense
Provides utilities to load and access configuration settings from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
import re


class ConfigLoader:
    """
    Loads and manages configuration from config.yaml file.
    Supports variable interpolation using ${section.key} syntax.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the config.yaml file (default: "config.yaml")
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_variables()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file or provide a valid path."
            )

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {self.config_path}")

        return config

    def _resolve_variables(self):
        """Resolve variable interpolation in the config (${section.key})."""
        self.config = self._resolve_dict(self.config, self.config)

    def _resolve_dict(self, obj: Any, root: Dict) -> Any:
        """Recursively resolve variables in dictionaries and strings."""
        if isinstance(obj, dict):
            return {k: self._resolve_dict(v, root) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_dict(item, root) for item in obj]
        elif isinstance(obj, str):
            return self._resolve_string(obj, root)
        else:
            return obj

    def _resolve_string(self, value: str, root: Dict) -> str:
        """Resolve variable references in a string."""
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        for match in matches:
            keys = match.split('.')
            resolved_value = root

            try:
                for key in keys:
                    resolved_value = resolved_value[key]
                value = value.replace(f'${{{match}}}', str(resolved_value))
            except (KeyError, TypeError):
                # Keep the original placeholder if resolution fails
                pass

        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Path to the config key (e.g., "eeg.preprocessing.max_sampling_rate")
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Example:
            >>> config = ConfigLoader()
            >>> sampling_rate = config.get("eeg.preprocessing.max_sampling_rate")
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_path(self, key_path: str, create_if_missing: bool = False) -> Path:
        """
        Get a path from configuration and optionally create it.

        Args:
            key_path: Path to the config key
            create_if_missing: Whether to create the directory if it doesn't exist

        Returns:
            Path object

        Example:
            >>> config = ConfigLoader()
            >>> output_path = config.get_path("paths.output.features", create_if_missing=True)
        """
        path_str = self.get(key_path)
        if path_str is None:
            raise ValueError(f"Path not found in config: {key_path}")

        path = Path(path_str)

        if create_if_missing and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_all_paths(self) -> Dict[str, Path]:
        """Get all configured paths as Path objects."""
        paths_config = self.get("paths", {})
        return self._flatten_paths(paths_config)

    def _flatten_paths(self, obj: Any, prefix: str = "") -> Dict[str, Path]:
        """Recursively flatten nested path dictionaries."""
        result = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(self._flatten_paths(value, new_prefix))
                elif isinstance(value, str):
                    result[new_prefix] = Path(value)

        return result

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"


# Singleton instance for easy import
_config_instance = None


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Get or create a singleton configuration instance.

    Args:
        config_path: Path to config file (default: "config.yaml")

    Returns:
        ConfigLoader instance

    Example:
        >>> from config_loader import get_config
        >>> config = get_config()
        >>> sampling_rate = config.get("eeg.preprocessing.max_sampling_rate")
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)

    return _config_instance


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = ConfigLoader()

    # Access configuration values
    print("EEG Max Sampling Rate:", config.get("eeg.preprocessing.max_sampling_rate"))
    print("Output Features Path:", config.get("paths.output.features"))

    # Get paths
    features_path = config.get_path("paths.output.features", create_if_missing=False)
    print(f"Features path: {features_path}")

    # Get all paths
    all_paths = config.get_all_paths()
    print("\nAll configured paths:")
    for path_name, path_value in all_paths.items():
        print(f"  {path_name}: {path_value}")
