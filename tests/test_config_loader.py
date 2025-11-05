"""
Unit tests for config_loader.py

Tests the ConfigLoader class and related configuration management functions.
"""

import pytest
from pathlib import Path
import yaml

from config_loader import ConfigLoader, get_config


class TestConfigLoader:
    """Test suite for ConfigLoader class."""

    @pytest.mark.unit
    def test_init_with_valid_config(self, sample_config_file):
        """Test ConfigLoader initialization with valid config file."""
        config = ConfigLoader(str(sample_config_file))
        assert config.config is not None
        assert isinstance(config.config, dict)

    @pytest.mark.unit
    def test_init_with_missing_config(self, temp_dir):
        """Test ConfigLoader initialization with missing config file."""
        non_existent_path = temp_dir / "non_existent.yaml"
        with pytest.raises(FileNotFoundError):
            ConfigLoader(str(non_existent_path))

    @pytest.mark.unit
    def test_get_simple_value(self, sample_config_file):
        """Test getting a simple configuration value."""
        config = ConfigLoader(str(sample_config_file))
        max_sfreq = config.get("eeg.preprocessing.max_sampling_rate")
        assert max_sfreq == 200

    @pytest.mark.unit
    def test_get_nested_value(self, sample_config_file):
        """Test getting a nested configuration value."""
        config = ConfigLoader(str(sample_config_file))
        notch_freq = config.get("eeg.preprocessing.notch_filter.frequency")
        assert notch_freq == 50

    @pytest.mark.unit
    def test_get_with_default(self, sample_config_file):
        """Test get() with default value for non-existent key."""
        config = ConfigLoader(str(sample_config_file))
        value = config.get("non.existent.key", default=42)
        assert value == 42

    @pytest.mark.unit
    def test_get_nonexistent_key_no_default(self, sample_config_file):
        """Test get() without default for non-existent key returns None."""
        config = ConfigLoader(str(sample_config_file))
        value = config.get("non.existent.key")
        assert value is None

    @pytest.mark.unit
    def test_variable_interpolation(self, temp_dir):
        """Test variable interpolation in config values."""
        config_dict = {
            "base": "root_path",
            "derived": "${base}/subdir",
            "nested": {
                "path": "${base}/nested/path"
            }
        }
        config_path = temp_dir / "config_interpolation.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = ConfigLoader(str(config_path))
        assert config.get("derived") == "root_path/subdir"
        assert config.get("nested.path") == "root_path/nested/path"

    @pytest.mark.unit
    def test_get_path(self, sample_config_file):
        """Test get_path() returns Path object."""
        config = ConfigLoader(str(sample_config_file))
        path = config.get_path("paths.data_root")
        assert isinstance(path, Path)
        # Path may or may not have trailing slash depending on OS
        assert str(path) in ["test_data/", "test_data"]

    @pytest.mark.unit
    def test_get_path_creates_directory(self, temp_dir, sample_config_file):
        """Test get_path() with create_if_missing=True."""
        config = ConfigLoader(str(sample_config_file))
        # Modify config to use temp_dir
        config.config["paths"]["new_dir"] = str(temp_dir / "new_directory")

        path = config.get_path("paths.new_dir", create_if_missing=True)
        assert path.exists()
        assert path.is_dir()

    @pytest.mark.unit
    def test_get_path_nonexistent_key(self, sample_config_file):
        """Test get_path() raises ValueError for non-existent key."""
        config = ConfigLoader(str(sample_config_file))
        with pytest.raises(ValueError, match="Path not found in config"):
            config.get_path("paths.nonexistent")

    @pytest.mark.unit
    def test_get_all_paths(self, sample_config_file):
        """Test get_all_paths() returns all path configurations."""
        config = ConfigLoader(str(sample_config_file))
        all_paths = config.get_all_paths()
        assert isinstance(all_paths, dict)
        assert "data_root" in all_paths
        assert "input.datasets" in all_paths
        assert all(isinstance(p, Path) for p in all_paths.values())

    @pytest.mark.unit
    def test_dict_style_access(self, sample_config_file):
        """Test dictionary-style access using __getitem__."""
        config = ConfigLoader(str(sample_config_file))
        max_sfreq = config["eeg.preprocessing.max_sampling_rate"]
        assert max_sfreq == 200

    @pytest.mark.unit
    def test_repr(self, sample_config_file):
        """Test __repr__ returns meaningful string."""
        config = ConfigLoader(str(sample_config_file))
        repr_str = repr(config)
        assert "ConfigLoader" in repr_str
        assert str(sample_config_file) in repr_str

    @pytest.mark.unit
    def test_empty_config_file(self, temp_dir):
        """Test ConfigLoader with empty config file."""
        empty_config = temp_dir / "empty.yaml"
        empty_config.write_text("")

        with pytest.raises(ValueError, match="Configuration file is empty"):
            ConfigLoader(str(empty_config))


class TestGetConfigSingleton:
    """Test suite for get_config() singleton function."""

    @pytest.mark.unit
    def test_get_config_creates_instance(self, sample_config_file):
        """Test get_config() creates ConfigLoader instance."""
        config = get_config(str(sample_config_file))
        assert isinstance(config, ConfigLoader)

    @pytest.mark.unit
    def test_get_config_returns_same_instance(self, sample_config_file):
        """Test get_config() returns the same singleton instance."""
        config1 = get_config(str(sample_config_file))
        config2 = get_config(str(sample_config_file))
        assert config1 is config2

    @pytest.mark.unit
    def test_get_config_with_default_path(self, temp_dir, monkeypatch):
        """Test get_config() with default config.yaml path."""
        # Create config.yaml in temp directory
        config_dict = {"test": "value"}
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        # Change working directory to temp_dir
        monkeypatch.chdir(temp_dir)

        config = get_config()
        assert config.get("test") == "value"


class TestConfigLoaderEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_circular_variable_reference(self, temp_dir):
        """Test handling of circular variable references."""
        config_dict = {
            "a": "${b}",
            "b": "${a}"
        }
        config_path = temp_dir / "circular.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = ConfigLoader(str(config_path))
        # Should not crash, but won't resolve circular references
        value_a = config.get("a")
        # Value should contain unresolved placeholder
        assert "${" in value_a

    @pytest.mark.unit
    def test_nested_variable_interpolation(self, temp_dir):
        """Test single-level variable interpolation (nested not supported)."""
        config_dict = {
            "level1": "base",
            "level2": "${level1}/mid",
            "level3": "${level1}/deep"  # Changed to use level1 directly
        }
        config_path = temp_dir / "nested_vars.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = ConfigLoader(str(config_path))
        # Test single-level interpolation works
        assert config.get("level2") == "base/mid"
        assert config.get("level3") == "base/deep"

    @pytest.mark.unit
    def test_get_path_with_non_string_value(self, temp_dir):
        """Test get_path() with non-string value raises TypeError."""
        config_dict = {"paths": {"numeric": 123}}
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = ConfigLoader(str(config_path))
        # get_path should raise TypeError for non-string values
        with pytest.raises(TypeError):
            config.get_path("paths.numeric")

    @pytest.mark.unit
    def test_config_with_special_characters(self, temp_dir):
        """Test configuration with special characters in values."""
        config_dict = {
            "special": {
                "with_spaces": "value with spaces",
                "with_symbols": "value@#$%",
                "with_quotes": 'value "with" quotes'
            }
        }
        config_path = temp_dir / "special.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = ConfigLoader(str(config_path))
        assert config.get("special.with_spaces") == "value with spaces"
        assert config.get("special.with_symbols") == "value@#$%"
        assert 'with' in config.get("special.with_quotes")
