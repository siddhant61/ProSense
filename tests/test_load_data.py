"""
Unit tests for load_data.py

Tests the LoadData class validation methods and data loading functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import mne

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

class TestLoadDataSignalProcessing:
    """Test suite for signal processing methods."""

    @pytest.fixture
    def loader(self):
        """Create LoadData instance."""
        return LoadData()

    @pytest.mark.unit
    def test_detect_spikes_basic(self, loader):
        """Test basic spike detection."""
        # Create data with a clear spike in the middle
        data = np.zeros((100, 3))  # 100 samples, 3 channels
        data[40:60, :] = 5.0  # Add spike with magnitude > threshold

        start_time, end_time = loader.detect_spikes(data, threshold=2.5)

        # Spike should be detected
        assert start_time >= 40
        assert end_time <= 60
        assert start_time <= end_time

    @pytest.mark.unit
    def test_detect_spikes_with_noise(self, loader):
        """Test spike detection with noisy data."""
        # Create data with noise and a clear spike
        np.random.seed(42)
        data = np.random.randn(100, 3) * 0.5  # Low noise
        data[50:55, :] = 10.0  # Add strong spike

        start_time, end_time = loader.detect_spikes(data, threshold=3.0)

        # Spike should be detected around the spike region
        assert 45 <= start_time <= 55
        assert 50 <= end_time <= 60

    @pytest.mark.unit
    def test_detect_spikes_custom_threshold(self, loader):
        """Test spike detection with custom threshold."""
        data = np.zeros((100, 3))
        data[30:40, :] = 2.0  # Moderate spike

        # Lower threshold should detect it
        start_time, end_time = loader.detect_spikes(data, threshold=1.0)
        assert start_time >= 30
        assert end_time <= 40

    @pytest.mark.unit
    def test_slice_signals_basic(self, loader):
        """Test basic signal slicing."""
        # Create sample dataset
        dataset = {
            'signal1': np.arange(100),
            'signal2': np.arange(100) * 2,
            'signal3': np.arange(100) * 3
        }

        # Slice from 20 to 80
        sliced = loader.slice_signals(dataset, 20, 80)

        # Check structure
        assert 'signal1' in sliced
        assert 'signal2' in sliced
        assert 'signal3' in sliced

        # Check lengths
        assert len(sliced['signal1']) == 60
        assert len(sliced['signal2']) == 60
        assert len(sliced['signal3']) == 60

        # Check values
        assert sliced['signal1'][0] == 20
        assert sliced['signal1'][-1] == 79
        assert sliced['signal2'][0] == 40
        assert sliced['signal3'][0] == 60

    @pytest.mark.unit
    def test_slice_signals_preserves_all_keys(self, loader):
        """Test that slice_signals preserves all dataset keys."""
        dataset = {
            'acc_x': np.arange(200),
            'acc_y': np.arange(200) + 100,
            'acc_z': np.arange(200) + 200,
            'gyro_x': np.arange(200) * 0.1,
            'gyro_y': np.arange(200) * 0.2,
        }

        sliced = loader.slice_signals(dataset, 50, 150)

        # All keys should be preserved
        assert set(sliced.keys()) == set(dataset.keys())

        # All sliced arrays should have same length
        lengths = [len(v) for v in sliced.values()]
        assert all(l == 100 for l in lengths)

    @pytest.mark.integration
    def test_detect_spikes_and_slice_pipeline(self, loader):
        """Test combined spike detection and signal slicing workflow."""
        # Create realistic accelerometer-like data
        np.random.seed(42)
        n_samples = 1000

        # Create dataset with multiple signals
        dataset = {
            'acc_x': np.random.randn(n_samples) * 0.5,
            'acc_y': np.random.randn(n_samples) * 0.5,
            'acc_z': 9.8 + np.random.randn(n_samples) * 0.5
        }

        # Add synchronized spike across all channels
        spike_start = 400
        spike_end = 450
        for key in dataset:
            dataset[key][spike_start:spike_end] += 5.0

        # Detect spikes using magnitude
        data_array = np.column_stack([dataset['acc_x'], dataset['acc_y'], dataset['acc_z']])
        start_time, end_time = loader.detect_spikes(data_array, threshold=3.0)

        # Slice signals based on detected spikes (with some padding)
        padding = 50
        sliced = loader.slice_signals(
            dataset,
            max(0, start_time - padding),
            min(n_samples, end_time + padding)
        )

        # Verify sliced region contains the spike
        assert all(len(v) > 0 for v in sliced.values())
        # Verify all channels have same length
        lengths = [len(v) for v in sliced.values()]
        assert len(set(lengths)) == 1  # All same length


class TestLoadDataMneConversion:
    """Test suite for MNE conversion functions."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance."""
        return LoadData()

    @pytest.mark.unit
    def test_convert_to_mne_basic(self, loader):
        """Test basic data conversion to MNE format."""
        # Create sample EEG data (4 channels, 1000 samples)
        n_channels = 4
        n_samples = 1000
        sfreq = 256.0

        data = np.random.randn(n_channels, n_samples)

        # Convert to MNE
        mne_raw = loader.convert_to_mne(data, sfreq)

        # Verify it's an MNE Raw object
        assert isinstance(mne_raw, mne.io.RawArray)

        # Verify channel names
        assert mne_raw.ch_names == ["AF7", "AF8", "TP9", "TP10"]

        # Verify sampling frequency
        assert mne_raw.info['sfreq'] == sfreq

        # Verify data shape
        assert mne_raw.get_data().shape == (n_channels, n_samples)

    @pytest.mark.unit
    def test_convert_to_mne_data_preservation(self, loader):
        """Test that data values are preserved during MNE conversion."""
        n_channels = 4
        n_samples = 100
        sfreq = 256.0

        # Create data with known values
        data = np.array([
            [1.0] * n_samples,  # AF7: all 1.0
            [2.0] * n_samples,  # AF8: all 2.0
            [3.0] * n_samples,  # TP9: all 3.0
            [4.0] * n_samples   # TP10: all 4.0
        ])

        mne_raw = loader.convert_to_mne(data, sfreq)
        converted_data = mne_raw.get_data()

        # Verify data values are preserved
        np.testing.assert_array_almost_equal(converted_data, data)


class TestLoadDataFormatDataset:
    """Test suite for format_dataset function."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance."""
        return LoadData()

    @pytest.fixture
    def sample_pickle_file(self, tmp_path):
        """Create a sample pickle file with EEG data."""
        # Create sample DataFrame with proper structure
        n_samples = 1000
        sfreq = 250.0  # 250 Hz

        # Create numeric timestamps (seconds)
        timestamps = np.arange(n_samples) / sfreq

        df = pd.DataFrame({
            'RAW_AF7': np.random.randn(n_samples),
            'RAW_AF8': np.random.randn(n_samples),
            'RAW_TP9': np.random.randn(n_samples),
            'RAW_TP10': np.random.randn(n_samples),
            'R_AUX': np.zeros(n_samples)  # Auxiliary channel to be dropped
        }, index=timestamps)

        # Save to pickle file
        pkl_file = tmp_path / "test_eeg_data.pkl"
        df.to_pickle(pkl_file)

        return pkl_file

    @pytest.mark.unit
    def test_format_dataset_basic(self, loader, sample_pickle_file):
        """Test basic dataset formatting from pickle file."""
        dataset = loader.format_dataset(str(sample_pickle_file))

        # Check structure
        assert isinstance(dataset, dict)
        assert len(dataset) == 1

        # Get the dataset entry
        dataset_name = list(dataset.keys())[0]
        assert dataset_name == "test_eeg_data"

        # Check dataset contents
        assert 'data' in dataset[dataset_name]
        assert 'sfreq' in dataset[dataset_name]

        # Verify MNE object
        assert isinstance(dataset[dataset_name]['data'], mne.io.RawArray)

        # Verify channel names (R_AUX should be dropped)
        assert len(dataset[dataset_name]['data'].ch_names) == 4
        assert set(dataset[dataset_name]['data'].ch_names) == {'AF7', 'AF8', 'TP9', 'TP10'}

    @pytest.mark.unit
    def test_format_dataset_drops_aux_channel(self, loader, tmp_path):
        """Test that R_AUX channel is dropped during formatting."""
        # Create DataFrame with R_AUX
        n_samples = 500
        sfreq = 250.0
        timestamps = np.arange(n_samples) / sfreq

        df = pd.DataFrame({
            'RAW_AF7': np.random.randn(n_samples),
            'RAW_AF8': np.random.randn(n_samples),
            'RAW_TP9': np.random.randn(n_samples),
            'RAW_TP10': np.random.randn(n_samples),
            'R_AUX': np.random.randn(n_samples)
        }, index=timestamps)

        pkl_file = tmp_path / "test_with_aux.pkl"
        df.to_pickle(pkl_file)

        dataset = loader.format_dataset(str(pkl_file))

        # Verify R_AUX was dropped
        dataset_name = list(dataset.keys())[0]
        mne_data = dataset[dataset_name]['data']
        assert 'R_AUX' not in mne_data.ch_names
        assert len(mne_data.ch_names) == 4

    @pytest.mark.unit
    def test_format_dataset_calculates_sfreq(self, loader, sample_pickle_file):
        """Test that sampling frequency is calculated correctly."""
        dataset = loader.format_dataset(str(sample_pickle_file))

        dataset_name = list(dataset.keys())[0]
        sfreq = dataset[dataset_name]['sfreq']

        # Should be approximately 250 Hz (4ms intervals)
        assert isinstance(sfreq, (int, float))
        assert 240 < sfreq < 260  # Allow some tolerance

    @pytest.mark.unit
    def test_format_dataset_invalid_not_dataframe(self, loader, tmp_path):
        """Test that format_dataset raises error for non-DataFrame data."""
        # Save a non-DataFrame to pickle
        invalid_file = tmp_path / "invalid.pkl"
        with open(invalid_file, 'wb') as f:
            pickle.dump({'not': 'a dataframe'}, f)

        with pytest.raises(ValueError, match="not a pandas DataFrame"):
            loader.format_dataset(str(invalid_file))

    @pytest.mark.unit
    def test_format_dataset_unnamed_columns(self, loader, tmp_path):
        """Test format_dataset with unnamed (numeric) columns."""
        # Create DataFrame with unnamed columns (RangeIndex)
        n_samples = 500
        sfreq = 250.0
        timestamps = np.arange(n_samples) / sfreq

        # Create DataFrame without named columns
        df = pd.DataFrame(
            np.random.randn(n_samples, 5),  # 5 columns: 4 EEG + 1 AUX
            index=timestamps
        )

        pkl_file = tmp_path / "unnamed_cols.pkl"
        df.to_pickle(pkl_file)

        dataset = loader.format_dataset(str(pkl_file))

        # Should successfully name columns and process
        dataset_name = list(dataset.keys())[0]
        mne_data = dataset[dataset_name]['data']
        assert len(mne_data.ch_names) == 4

    @pytest.mark.unit
    def test_format_dataset_wrong_column_count(self, loader, tmp_path):
        """Test format_dataset raises error for wrong number of columns."""
        # Create DataFrame with wrong number of columns (not 5)
        n_samples = 500
        sfreq = 250.0
        timestamps = np.arange(n_samples) / sfreq

        df = pd.DataFrame(
            np.random.randn(n_samples, 3),  # Wrong: only 3 columns instead of 5
            index=timestamps
        )

        pkl_file = tmp_path / "wrong_cols.pkl"
        df.to_pickle(pkl_file)

        with pytest.raises(ValueError, match="columns.*expected"):
            loader.format_dataset(str(pkl_file))

    @pytest.mark.unit
    def test_format_dataset_wrong_column_names(self, loader, tmp_path):
        """Test format_dataset raises error for wrong column names."""
        # Create DataFrame with wrong column names
        n_samples = 500
        sfreq = 250.0
        timestamps = np.arange(n_samples) / sfreq

        df = pd.DataFrame({
            'WRONG_1': np.random.randn(n_samples),
            'WRONG_2': np.random.randn(n_samples),
            'WRONG_3': np.random.randn(n_samples),
            'WRONG_4': np.random.randn(n_samples),
        }, index=timestamps)

        pkl_file = tmp_path / "wrong_names.pkl"
        df.to_pickle(pkl_file)

        with pytest.raises(ValueError, match="don't match expected channels"):
            loader.format_dataset(str(pkl_file))

    @pytest.mark.unit
    def test_format_dataset_none_sfreq(self, loader, tmp_path):
        """Test format_dataset raises error when timestamps are invalid."""
        # Create DataFrame with invalid timestamps (all same value)
        n_samples = 500
        timestamps = np.zeros(n_samples)  # All zeros - not monotonically increasing

        df = pd.DataFrame({
            'RAW_AF7': np.random.randn(n_samples),
            'RAW_AF8': np.random.randn(n_samples),
            'RAW_TP9': np.random.randn(n_samples),
            'RAW_TP10': np.random.randn(n_samples),
        }, index=timestamps)

        pkl_file = tmp_path / "invalid_ts.pkl"
        df.to_pickle(pkl_file)

        # Should raise error about non-monotonic timestamps
        with pytest.raises(ValueError, match="Timestamps are not monotonically increasing"):
            loader.format_dataset(str(pkl_file))


class TestLoadDataFifOperations:
    """Test suite for FIF file save/load operations."""

    @pytest.fixture
    def loader(self):
        """Create a LoadData instance."""
        return LoadData()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with MNE data."""
        n_channels = 4
        n_samples = 1000
        sfreq = 256.0

        # Create MNE Raw data
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=["AF7", "AF8", "TP9", "TP10"],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info)

        # Create dataset dictionary
        # Note: Key needs exactly one underscore (load_fif_dataset extracts up to 2nd underscore)
        dataset = {
            'test_stream1': {
                'data': raw,
                'sfreq': sfreq
            }
        }
        return dataset

    @pytest.fixture
    def sample_dataset_with_epochs(self):
        """Create a sample dataset with both Raw and Epochs."""
        n_channels = 4
        n_samples = 1000
        sfreq = 256.0

        # Create MNE Raw data
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=["AF7", "AF8", "TP9", "TP10"],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info)

        # Create events for epochs
        events = np.array([[100, 0, 1], [300, 0, 1], [500, 0, 1]])
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None, preload=True)

        # Create dataset dictionary
        # Note: Key needs exactly one underscore (load_fif_dataset extracts up to 2nd underscore)
        dataset = {
            'test_stream2': {
                'data': raw,
                'epochs': epochs,
                'sfreq': sfreq
            }
        }
        return dataset

    @pytest.mark.integration
    def test_save_fif_dataset_basic(self, loader, sample_dataset, tmp_path):
        """Test saving dataset to FIF format."""
        save_dir = tmp_path / "fif_output"
        save_dir.mkdir()

        # Save the dataset
        loader.save_fif_dataset(sample_dataset, str(save_dir))

        # Check that files were created
        assert (save_dir / "test_stream1_raw.fif").exists()
        assert (save_dir / "test_stream1_sfreq.txt").exists()

        # Verify sfreq was saved correctly
        with open(save_dir / "test_stream1_sfreq.txt", 'r') as f:
            saved_sfreq = float(f.read())
            assert saved_sfreq == 256.0

    @pytest.mark.integration
    def test_save_fif_dataset_with_epochs(self, loader, sample_dataset_with_epochs, tmp_path):
        """Test saving dataset with epochs to FIF format."""
        save_dir = tmp_path / "fif_output_epochs"
        save_dir.mkdir()

        # Save the dataset
        loader.save_fif_dataset(sample_dataset_with_epochs, str(save_dir))

        # Check that all files were created
        assert (save_dir / "test_stream2_raw.fif").exists()
        assert (save_dir / "test_stream2_epo.fif").exists()
        assert (save_dir / "test_stream2_sfreq.txt").exists()

    @pytest.mark.integration
    def test_load_fif_dataset_basic(self, loader, sample_dataset, tmp_path):
        """Test loading dataset from FIF format."""
        save_dir = tmp_path / "fif_for_load"
        save_dir.mkdir()

        # Save first
        loader.save_fif_dataset(sample_dataset, str(save_dir))

        # Load back
        loaded_dataset = loader.load_fif_dataset(str(save_dir))

        # Check structure
        assert isinstance(loaded_dataset, dict)
        assert 'test_stream1' in loaded_dataset
        assert 'data' in loaded_dataset['test_stream1']
        assert 'sfreq' in loaded_dataset['test_stream1']

        # Verify data type
        assert isinstance(loaded_dataset['test_stream1']['data'], mne.io.Raw)
        assert loaded_dataset['test_stream1']['sfreq'] == 256.0

    @pytest.mark.integration
    def test_load_fif_dataset_with_epochs(self, loader, sample_dataset_with_epochs, tmp_path):
        """Test loading dataset with epochs from FIF format."""
        save_dir = tmp_path / "fif_for_load_epochs"
        save_dir.mkdir()

        # Save first
        loader.save_fif_dataset(sample_dataset_with_epochs, str(save_dir))

        # Load back
        loaded_dataset = loader.load_fif_dataset(str(save_dir))

        # Check structure
        assert isinstance(loaded_dataset, dict)
        assert 'test_stream2' in loaded_dataset
        assert 'data' in loaded_dataset['test_stream2']
        assert 'epochs' in loaded_dataset['test_stream2']
        assert 'sfreq' in loaded_dataset['test_stream2']

        # Verify data types
        assert isinstance(loaded_dataset['test_stream2']['data'], mne.io.Raw)
        # Epochs loaded from FIF are EpochsFIF, check for BaseEpochs instead
        assert isinstance(loaded_dataset['test_stream2']['epochs'], mne.BaseEpochs)
        assert loaded_dataset['test_stream2']['sfreq'] == 256.0

    @pytest.mark.integration
    def test_save_and_load_cycle_preserves_data(self, loader, sample_dataset, tmp_path):
        """Test that save and load cycle preserves data accurately."""
        save_dir = tmp_path / "fif_cycle_test"
        save_dir.mkdir()

        # Get original data
        original_data = sample_dataset['test_stream1']['data'].get_data()

        # Save
        loader.save_fif_dataset(sample_dataset, str(save_dir))

        # Load
        loaded_dataset = loader.load_fif_dataset(str(save_dir))

        # Get loaded data
        loaded_data = loaded_dataset['test_stream1']['data'].get_data()

        # Verify data is preserved
        np.testing.assert_array_almost_equal(original_data, loaded_data, decimal=6)
