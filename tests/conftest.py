"""
Pytest configuration and shared fixtures for ProSense tests.

This file contains fixtures that are available to all tests in the test suite.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict() -> dict:
    """
    Sample configuration dictionary for testing.

    Returns:
        Dictionary with sample configuration
    """
    return {
        "paths": {
            "data_root": "test_data/",
            "input": {
                "datasets": "test_data/datasets/",
                "logs": "test_data/logs/",
            },
            "output": {
                "base": "test_data/output/",
                "features": "test_data/output/features/",
            }
        },
        "eeg": {
            "preprocessing": {
                "max_sampling_rate": 200,
                "notch_filter": {"frequency": 50},
                "bandpass_filter": {"low_freq": 1.0, "high_freq": 40.0},
                "epoch_duration": 5.0,
            }
        },
        "logging": {
            "level": "INFO",
        }
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config_dict: dict) -> Path:
    """
    Create a temporary config.yaml file for testing.

    Args:
        temp_dir: Temporary directory fixture
        sample_config_dict: Sample configuration dictionary

    Returns:
        Path to the created config.yaml file
    """
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Create a sample DataFrame for testing.

    Returns:
        DataFrame with sample physiological data
    """
    timestamps = pd.date_range('2024-01-01', periods=100, freq='10ms')
    data = {
        'timestamp': timestamps,
        'channel_1': np.random.randn(100),
        'channel_2': np.random.randn(100),
        'channel_3': np.random.randn(100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_timestamps() -> np.ndarray:
    """
    Create sample timestamps for testing.

    Returns:
        Numpy array of Unix timestamps
    """
    start_time = 1704067200  # 2024-01-01 00:00:00 UTC
    timestamps = np.linspace(start_time, start_time + 10, 100)
    return timestamps


@pytest.fixture
def sample_eeg_data() -> np.ndarray:
    """
    Create sample EEG data for testing.

    Returns:
        Numpy array shaped (n_channels, n_samples)
    """
    n_channels = 4
    n_samples = 1000
    # Generate synthetic EEG-like data with typical frequency components
    data = np.random.randn(n_channels, n_samples) * 50  # µV scale
    return data


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_dataframe: pd.DataFrame) -> Path:
    """
    Create a temporary CSV file with sample data.

    Args:
        temp_dir: Temporary directory fixture
        sample_dataframe: Sample DataFrame to save

    Returns:
        Path to the created CSV file
    """
    csv_path = temp_dir / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_pickle_file(temp_dir: Path) -> Path:
    """
    Create a temporary pickle file with sample data.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the created pickle file
    """
    import pickle

    pickle_path = temp_dir / "sample_data.pkl"
    sample_data = {
        'dataset_1': {
            'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
            'sfreq': 100,
        },
        'dataset_2': {
            'data': pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]}),
            'sfreq': 200,
        }
    }

    with open(pickle_path, 'wb') as f:
        pickle.dump(sample_data, f)

    return pickle_path


@pytest.fixture
def invalid_pickle_file(temp_dir: Path) -> Path:
    """
    Create a file with .pkl extension but invalid pickle content.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the invalid pickle file
    """
    pickle_path = temp_dir / "invalid.pkl"
    with open(pickle_path, 'w') as f:
        f.write("This is not a valid pickle file")
    return pickle_path


@pytest.fixture
def empty_directory(temp_dir: Path) -> Path:
    """
    Create an empty directory for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to empty directory
    """
    empty_dir = temp_dir / "empty_dir"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """
    Reset the config singleton between tests.

    This ensures each test gets a fresh configuration instance.
    """
    import config_loader
    config_loader._config_instance = None
    yield
    config_loader._config_instance = None


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring actual data files"
    )
