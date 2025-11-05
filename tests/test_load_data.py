"""
Unit tests for load_data.py

Tests the LoadData class validation methods and data loading functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from load_data import LoadData


class TestLoadDataValidation:
    """Test suite for LoadData validation methods."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance for testing."""
        return LoadData()

    # =============================================================================
    # TEST validate_file_exists()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_file_exists_valid(self, loader, sample_csv_file):
        """Test validate_file_exists() with valid file."""
        assert loader.validate_file_exists(str(sample_csv_file)) is True

    @pytest.mark.unit
    def test_validate_file_exists_missing(self, loader, temp_dir):
        """Test validate_file_exists() with missing file."""
        non_existent = temp_dir / "missing.csv"
        with pytest.raises(FileNotFoundError):
            loader.validate_file_exists(str(non_existent))

    @pytest.mark.unit
    def test_validate_file_exists_empty_path(self, loader):
        """Test validate_file_exists() with empty path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            loader.validate_file_exists("")

    @pytest.mark.unit
    def test_validate_file_exists_directory(self, loader, temp_dir):
        """Test validate_file_exists() rejects directory."""
        with pytest.raises(ValueError, match="Path is not a file"):
            loader.validate_file_exists(str(temp_dir))

    # =============================================================================
    # TEST validate_directory_exists()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_directory_exists_valid(self, loader, temp_dir):
        """Test validate_directory_exists() with valid directory."""
        assert loader.validate_directory_exists(str(temp_dir)) is True

    @pytest.mark.unit
    def test_validate_directory_exists_missing(self, loader, temp_dir):
        """Test validate_directory_exists() with missing directory."""
        non_existent = temp_dir / "missing_dir"
        with pytest.raises(FileNotFoundError):
            loader.validate_directory_exists(str(non_existent))

    @pytest.mark.unit
    def test_validate_directory_exists_empty_path(self, loader):
        """Test validate_directory_exists() with empty path."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            loader.validate_directory_exists("")

    @pytest.mark.unit
    def test_validate_directory_exists_file(self, loader, sample_csv_file):
        """Test validate_directory_exists() rejects file."""
        with pytest.raises(ValueError, match="Path is not a directory"):
            loader.validate_directory_exists(str(sample_csv_file))

    # =============================================================================
    # TEST validate_file_extension()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_file_extension_valid(self, loader, sample_csv_file):
        """Test validate_file_extension() with correct extension."""
        assert loader.validate_file_extension(str(sample_csv_file), '.csv') is True

    @pytest.mark.unit
    def test_validate_file_extension_wrong(self, loader, sample_csv_file):
        """Test validate_file_extension() with wrong extension."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            loader.validate_file_extension(str(sample_csv_file), '.pkl')

    @pytest.mark.unit
    def test_validate_file_extension_case_insensitive(self, loader, temp_dir):
        """Test validate_file_extension() is case insensitive."""
        file_path = temp_dir / "data.CSV"
        file_path.touch()
        assert loader.validate_file_extension(str(file_path), '.csv') is True

    # =============================================================================
    # TEST validate_dataframe()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_dataframe_valid(self, loader, sample_dataframe):
        """Test validate_dataframe() with valid DataFrame."""
        assert loader.validate_dataframe(sample_dataframe) is True

    @pytest.mark.unit
    def test_validate_dataframe_wrong_type(self, loader):
        """Test validate_dataframe() rejects non-DataFrame."""
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            loader.validate_dataframe([1, 2, 3])

    @pytest.mark.unit
    def test_validate_dataframe_empty(self, loader):
        """Test validate_dataframe() rejects empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            loader.validate_dataframe(empty_df)

    @pytest.mark.unit
    def test_validate_dataframe_insufficient_rows(self, loader):
        """Test validate_dataframe() with min_rows requirement."""
        small_df = pd.DataFrame({'col1': [1, 2]})
        with pytest.raises(ValueError, match="insufficient data"):
            loader.validate_dataframe(small_df, min_rows=10)

    @pytest.mark.unit
    def test_validate_dataframe_missing_columns(self, loader, sample_dataframe):
        """Test validate_dataframe() with required columns."""
        with pytest.raises(ValueError, match="missing required columns"):
            loader.validate_dataframe(
                sample_dataframe,
                required_columns=['timestamp', 'missing_col']
            )

    @pytest.mark.unit
    def test_validate_dataframe_with_required_columns(self, loader, sample_dataframe):
        """Test validate_dataframe() with correct required columns."""
        assert loader.validate_dataframe(
            sample_dataframe,
            required_columns=['timestamp', 'channel_1']
        ) is True

    # =============================================================================
    # TEST validate_timestamps()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_timestamps_valid(self, loader, sample_timestamps):
        """Test validate_timestamps() with valid timestamps."""
        assert loader.validate_timestamps(sample_timestamps) is True

    @pytest.mark.unit
    def test_validate_timestamps_empty(self, loader):
        """Test validate_timestamps() rejects empty array."""
        with pytest.raises(ValueError, match="Timestamp array is empty"):
            loader.validate_timestamps(np.array([]))

    @pytest.mark.unit
    def test_validate_timestamps_with_nan(self, loader):
        """Test validate_timestamps() rejects NaN values."""
        timestamps = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="contain NaN or None values"):
            loader.validate_timestamps(timestamps)

    @pytest.mark.unit
    def test_validate_timestamps_non_numeric(self, loader):
        """Test validate_timestamps() rejects non-numeric values."""
        timestamps = pd.Series(['a', 'b', 'c'])
        with pytest.raises(ValueError, match="non-numeric values"):
            loader.validate_timestamps(timestamps)

    @pytest.mark.unit
    def test_validate_timestamps_not_monotonic(self, loader):
        """Test validate_timestamps() rejects non-monotonic timestamps."""
        timestamps = np.array([1.0, 3.0, 2.0, 4.0])  # Out of order
        with pytest.raises(ValueError, match="not monotonically increasing"):
            loader.validate_timestamps(timestamps)

    @pytest.mark.unit
    def test_validate_timestamps_monotonic_check_disabled(self, loader):
        """Test validate_timestamps() with monotonic check disabled."""
        timestamps = np.array([1.0, 3.0, 2.0, 4.0])
        # Should not raise when check_monotonic=False
        assert loader.validate_timestamps(timestamps, check_monotonic=False) is True

    # =============================================================================
    # TEST validate_sampling_rate()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_sampling_rate_valid(self, loader):
        """Test validate_sampling_rate() with valid rate."""
        assert loader.validate_sampling_rate(100) is True
        assert loader.validate_sampling_rate(250.5) is True

    @pytest.mark.unit
    def test_validate_sampling_rate_none(self, loader):
        """Test validate_sampling_rate() rejects None."""
        with pytest.raises(ValueError, match="Sampling rate cannot be None"):
            loader.validate_sampling_rate(None)

    @pytest.mark.unit
    def test_validate_sampling_rate_negative(self, loader):
        """Test validate_sampling_rate() rejects negative values."""
        with pytest.raises(ValueError, match="must be positive"):
            loader.validate_sampling_rate(-100)

    @pytest.mark.unit
    def test_validate_sampling_rate_zero(self, loader):
        """Test validate_sampling_rate() rejects zero."""
        with pytest.raises(ValueError, match="must be positive"):
            loader.validate_sampling_rate(0)

    @pytest.mark.unit
    def test_validate_sampling_rate_too_low(self, loader):
        """Test validate_sampling_rate() rejects too low rates."""
        with pytest.raises(ValueError, match="outside valid range"):
            loader.validate_sampling_rate(0.5, min_sfreq=1)

    @pytest.mark.unit
    def test_validate_sampling_rate_too_high(self, loader):
        """Test validate_sampling_rate() rejects too high rates."""
        with pytest.raises(ValueError, match="outside valid range"):
            loader.validate_sampling_rate(20000, max_sfreq=10000)

    @pytest.mark.unit
    def test_validate_sampling_rate_invalid_type(self, loader):
        """Test validate_sampling_rate() rejects invalid types."""
        with pytest.raises(ValueError, match="Invalid sampling rate type"):
            loader.validate_sampling_rate("invalid")

    # =============================================================================
    # TEST validate_data_array()
    # =============================================================================

    @pytest.mark.unit
    def test_validate_data_array_valid_numpy(self, loader):
        """Test validate_data_array() with valid numpy array."""
        data = np.random.randn(4, 1000)
        assert loader.validate_data_array(data) is True

    @pytest.mark.unit
    def test_validate_data_array_valid_dataframe(self, loader, sample_dataframe):
        """Test validate_data_array() with valid DataFrame."""
        assert loader.validate_data_array(sample_dataframe) is True

    @pytest.mark.unit
    def test_validate_data_array_none(self, loader):
        """Test validate_data_array() rejects None."""
        with pytest.raises(ValueError, match="Data array cannot be None"):
            loader.validate_data_array(None)

    @pytest.mark.unit
    def test_validate_data_array_wrong_type(self, loader):
        """Test validate_data_array() rejects wrong types."""
        with pytest.raises(TypeError, match="Expected numpy array or pandas object"):
            loader.validate_data_array([1, 2, 3])

    @pytest.mark.unit
    def test_validate_data_array_empty(self, loader):
        """Test validate_data_array() rejects empty array."""
        with pytest.raises(ValueError, match="Data array is empty"):
            loader.validate_data_array(np.array([]))

    @pytest.mark.unit
    def test_validate_data_array_insufficient_samples(self, loader):
        """Test validate_data_array() with min_samples requirement."""
        data = np.random.randn(4, 10)
        with pytest.raises(ValueError, match="Insufficient samples"):
            loader.validate_data_array(data, min_samples=100)

    @pytest.mark.unit
    def test_validate_data_array_too_many_channels(self, loader):
        """Test validate_data_array() with max_channels limit."""
        data = np.random.randn(20, 1000)
        with pytest.raises(ValueError, match="Too many channels"):
            loader.validate_data_array(data, max_channels=10)

    @pytest.mark.unit
    def test_validate_data_array_with_nan(self, loader):
        """Test validate_data_array() rejects NaN values."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(ValueError, match="Data contains NaN values"):
            loader.validate_data_array(data)

    @pytest.mark.unit
    def test_validate_data_array_with_inf(self, loader):
        """Test validate_data_array() rejects Inf values."""
        data = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(ValueError, match="Data contains Inf values"):
            loader.validate_data_array(data)


class TestLoadDataMethods:
    """Test suite for LoadData data loading methods."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance for testing."""
        return LoadData()

    # =============================================================================
    # TEST load_csv()
    # =============================================================================

    @pytest.mark.unit
    def test_load_csv_valid(self, loader, sample_csv_file):
        """Test load_csv() with valid CSV file."""
        data = loader.load_csv(str(sample_csv_file), ['timestamp', 'channel_1'])
        assert isinstance(data, pd.DataFrame)
        assert 'timestamp' in data.columns
        assert 'channel_1' in data.columns

    @pytest.mark.unit
    def test_load_csv_missing_file(self, loader, temp_dir):
        """Test load_csv() with missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load_csv(str(temp_dir / "missing.csv"), ['col1'])

    @pytest.mark.unit
    def test_load_csv_wrong_extension(self, loader, sample_pickle_file):
        """Test load_csv() rejects non-CSV files."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            loader.load_csv(str(sample_pickle_file), ['col1'])

    # =============================================================================
    # TEST load_pkl_dataset() and save_pkl_dataset()
    # =============================================================================

    @pytest.mark.unit
    def test_load_pkl_dataset_valid(self, loader, sample_pickle_file):
        """Test load_pkl_dataset() with valid pickle file."""
        data = loader.load_pkl_dataset(str(sample_pickle_file))
        assert isinstance(data, dict)
        assert 'dataset_1' in data
        assert 'dataset_2' in data

    @pytest.mark.unit
    def test_load_pkl_dataset_missing_file(self, loader, temp_dir):
        """Test load_pkl_dataset() with missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load_pkl_dataset(str(temp_dir / "missing.pkl"))

    @pytest.mark.unit
    def test_load_pkl_dataset_wrong_extension(self, loader, sample_csv_file):
        """Test load_pkl_dataset() rejects non-pickle files."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            loader.load_pkl_dataset(str(sample_csv_file))

    @pytest.mark.unit
    def test_load_pkl_dataset_invalid_content(self, loader, invalid_pickle_file):
        """Test load_pkl_dataset() with invalid pickle content."""
        with pytest.raises(ValueError, match="(Error loading pickle file|Failed to unpickle file)"):
            loader.load_pkl_dataset(str(invalid_pickle_file))

    @pytest.mark.unit
    def test_save_pkl_dataset_valid(self, loader, temp_dir):
        """Test save_pkl_dataset() creates valid pickle file."""
        data = {'key': 'value', 'number': 42}
        pkl_path = temp_dir / "output.pkl"

        loader.save_pkl_dataset(data, str(pkl_path))

        assert pkl_path.exists()
        # Verify we can load it back
        loaded = loader.load_pkl_dataset(str(pkl_path))
        assert loaded == data

    @pytest.mark.unit
    def test_save_pkl_dataset_none(self, loader, temp_dir):
        """Test save_pkl_dataset() rejects None data."""
        with pytest.raises(ValueError, match="Cannot save None data"):
            loader.save_pkl_dataset(None, str(temp_dir / "output.pkl"))

    @pytest.mark.unit
    def test_save_pkl_dataset_empty_path(self, loader):
        """Test save_pkl_dataset() rejects empty path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            loader.save_pkl_dataset({'data': 'value'}, "")

    @pytest.mark.unit
    def test_save_pkl_dataset_creates_parent_dir(self, loader, temp_dir):
        """Test save_pkl_dataset() creates parent directories."""
        nested_path = temp_dir / "subdir" / "nested" / "output.pkl"
        data = {'test': 'data'}

        loader.save_pkl_dataset(data, str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()

    # =============================================================================
    # TEST calculate_sfreq()
    # =============================================================================

    @pytest.mark.unit
    def test_calculate_sfreq_valid(self, loader):
        """Test calculate_sfreq() with valid timestamps."""
        # 100 Hz sampling rate
        timestamps = np.linspace(1704067200, 1704067200 + 1, 101)
        sfreq = loader.calculate_sfreq(timestamps)
        assert sfreq == 100

    @pytest.mark.unit
    def test_calculate_sfreq_invalid_timestamps(self, loader):
        """Test calculate_sfreq() rejects invalid timestamps."""
        timestamps = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError):
            loader.calculate_sfreq(timestamps)

    # =============================================================================
    # TEST process_datasets()
    # =============================================================================

    @pytest.mark.unit
    def test_process_datasets_missing_directory(self, loader, temp_dir):
        """Test process_datasets() with missing directory."""
        with pytest.raises(FileNotFoundError):
            loader.process_datasets(str(temp_dir / "missing"))

    @pytest.mark.unit
    def test_process_datasets_empty_directory(self, loader, empty_directory):
        """Test process_datasets() with empty directory."""
        output = loader.process_datasets(str(empty_directory))
        assert output.exists()
        assert output.is_dir()


class TestLoadDataIntegration:
    """Integration tests for LoadData class."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance for testing."""
        return LoadData()

    @pytest.mark.integration
    def test_save_and_load_cycle(self, loader, temp_dir):
        """Test complete save-load cycle."""
        original_data = {
            'dataset1': {
                'data': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                'sfreq': 100
            }
        }

        pkl_path = temp_dir / "test.pkl"

        # Save
        loader.save_pkl_dataset(original_data, str(pkl_path))

        # Load
        loaded_data = loader.load_pkl_dataset(str(pkl_path))

        # Verify
        assert 'dataset1' in loaded_data
        assert loaded_data['dataset1']['sfreq'] == 100
        pd.testing.assert_frame_equal(
            loaded_data['dataset1']['data'],
            original_data['dataset1']['data']
        )
