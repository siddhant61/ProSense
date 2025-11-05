"""
Unit tests for prepro_data.py

Tests the PreProData class and utility functions for data preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import mne
from prepro_data import PreProData, save_figures


class TestSaveFigures:
    """Test suite for save_figures utility function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_figures(self):
        """Create sample matplotlib figures."""
        figs = []
        titles = []

        # Create 2 simple figures
        for i in range(2):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            ax.set_title(f'Test Figure {i}')
            figs.append(fig)
            titles.append(f'Test Figure {i}')

        return figs, titles

    @pytest.mark.unit
    def test_save_figures_creates_directory(self, temp_dir, sample_figures):
        """Test that save_figures creates the target directory."""
        figs, titles = sample_figures
        name = 'test_plots'

        save_figures(figs, titles, name, temp_dir)

        # Check that directory was created
        expected_path = os.path.join(temp_dir, name)
        assert os.path.exists(expected_path)
        assert os.path.isdir(expected_path)

    @pytest.mark.unit
    def test_save_figures_saves_all_figures(self, temp_dir, sample_figures):
        """Test that all figures are saved."""
        figs, titles = sample_figures
        name = 'test_plots'

        save_figures(figs, titles, name, temp_dir)

        # Check that all files were created
        saved_files = os.listdir(os.path.join(temp_dir, name))
        assert len(saved_files) == 2
        assert all(f.endswith('.png') for f in saved_files)

    @pytest.mark.unit
    def test_save_figures_sanitizes_filenames(self, temp_dir):
        """Test that special characters in titles are replaced."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        # Title with special characters
        title = 'Test: Plot with/special\\characters?'
        figs = [fig]
        titles = [title]
        name = 'test_plots'

        save_figures(figs, titles, name, temp_dir)

        # Check that file was created with sanitized name
        saved_files = os.listdir(os.path.join(temp_dir, name))
        assert len(saved_files) == 1
        # Special chars should be replaced with underscores
        assert ':' not in saved_files[0]
        assert '/' not in saved_files[0]
        assert '\\' not in saved_files[0]


class TestPreProData:
    """Test suite for PreProData class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary dataset directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def prepro_instance(self, temp_dir):
        """Create PreProData instance."""
        return PreProData(temp_dir)

    @pytest.mark.unit
    def test_init(self, temp_dir):
        """Test PreProData initialization."""
        prepro = PreProData(temp_dir)
        assert prepro is not None
        assert prepro.dataset_folder == temp_dir

    @pytest.mark.unit
    def test_handle_nans_interpolates(self, prepro_instance):
        """Test that handle_nans interpolates NaN values."""
        # Create DataFrame with NaN values
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0]
        })

        result = prepro_instance.handle_nans(df)

        # Check that no NaNs remain
        assert not result.isnull().any().any()

        # Check interpolation worked correctly
        assert result['A'].iloc[1] == 2.0  # Linear interpolation: (1+3)/2
        assert result['B'].iloc[2] == 30.0  # Linear interpolation: (20+40)/2

    @pytest.mark.unit
    def test_handle_nans_fills_remaining(self, prepro_instance):
        """Test that handle_nans fills remaining NaNs."""
        # Create DataFrame with NaN at edges (can't interpolate)
        df = pd.DataFrame({
            'A': [np.nan, 2.0, 3.0],
            'B': [1.0, 2.0, np.nan]
        })

        result = prepro_instance.handle_nans(df)

        # Check that all NaNs are filled
        assert not result.isnull().any().any()
        # Leading NaNs filled with 0 by fillna
        assert result['A'].iloc[0] == 0.0
        # Trailing NaNs get forward-filled by interpolate, then fillna if needed
        assert result['B'].iloc[2] >= 0.0  # Either forward-filled or 0

    @pytest.mark.unit
    def test_trim_data_to_events_with_dataframe(self, prepro_instance):
        """Test trimming DataFrame to event time range."""
        # Create DataFrame with datetime index
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=100, freq='1s')
        df = pd.DataFrame({
            'value': range(100)
        }, index=timestamps)

        # Define start and end times
        start_time = pd.Timestamp('2024-01-01 10:00:10')
        end_time = pd.Timestamp('2024-01-01 10:00:20')

        result = prepro_instance.trim_data_to_events(df, start_time, end_time)

        # Check that data was trimmed correctly
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 11  # 10 seconds inclusive + start = 11 samples
        assert result.index[0] == start_time
        assert result.index[-1] == end_time

    @pytest.mark.unit
    def test_trim_data_to_events_empty_result(self, prepro_instance, capsys):
        """Test trimming with time range that yields empty result."""
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=100, freq='1s')
        df = pd.DataFrame({'value': range(100)}, index=timestamps)

        # Use time range outside data range
        start_time = pd.Timestamp('2024-01-01 11:00:00')
        end_time = pd.Timestamp('2024-01-01 11:00:10')

        result = prepro_instance.trim_data_to_events(df, start_time, end_time)

        # Check empty result and warning
        assert result.empty
        captured = capsys.readouterr()
        assert 'Warning' in captured.out

    @pytest.mark.unit
    def test_trim_data_to_events_converts_index(self, prepro_instance):
        """Test that non-datetime index is converted."""
        # Create DataFrame with string index
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        }, index=['2024-01-01 10:00:00', '2024-01-01 10:00:01',
                  '2024-01-01 10:00:02', '2024-01-01 10:00:03',
                  '2024-01-01 10:00:04'])

        start_time = pd.Timestamp('2024-01-01 10:00:01')
        end_time = pd.Timestamp('2024-01-01 10:00:03')

        result = prepro_instance.trim_data_to_events(df, start_time, end_time)

        # Check conversion and trimming
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3

    @pytest.mark.unit
    def test_adjust_marker_timestamps(self, prepro_instance):
        """Test adjusting marker timestamps by time shift."""
        # Create sample markers
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        markers = [
            ('Event1', base_time, 'task', timedelta(seconds=5)),
            ('Event2', base_time + timedelta(seconds=10), 'task', timedelta(seconds=5)),
            ('Event3', base_time + timedelta(seconds=20), 'task', timedelta(seconds=5))
        ]

        # Apply time shift of +2 seconds
        time_shift = timedelta(seconds=2)

        adjusted = prepro_instance.adjust_marker_timestamps(markers, time_shift)

        # Check that all markers were shifted
        assert len(adjusted) == 3
        assert adjusted[0][1] == base_time + timedelta(seconds=2)
        assert adjusted[1][1] == base_time + timedelta(seconds=12)
        assert adjusted[2][1] == base_time + timedelta(seconds=22)

        # Check other fields preserved
        assert adjusted[0][0] == 'Event1'
        assert adjusted[0][2] == 'task'
        assert adjusted[0][3] == timedelta(seconds=5)

    @pytest.mark.unit
    def test_calculate_time_shifts(self, prepro_instance):
        """Test calculating time shifts from detected vs original times."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Original markers with Sync Events
        markers = [
            ('SYNC1', base_time, 'Sync Events', timedelta(seconds=5)),
            ('Task1', base_time + timedelta(seconds=10), 'task', timedelta(seconds=5)),
            ('SYNC2', base_time + timedelta(seconds=20), 'Sync Events', timedelta(seconds=5))
        ]

        # Detected sync times (with 2 second delay)
        detected_sync_times = {
            base_time: base_time + timedelta(seconds=2),
            base_time + timedelta(seconds=20): base_time + timedelta(seconds=22)
        }

        time_shifts = prepro_instance.calculate_time_shifts(markers, detected_sync_times)

        # Check that time shifts were calculated
        assert len(time_shifts) == 2
        assert time_shifts[base_time] == timedelta(seconds=2)
        assert time_shifts[base_time + timedelta(seconds=20)] == timedelta(seconds=2)


class TestPreProDataIntegration:
    """Integration tests for PreProData preprocessing."""

    @pytest.fixture
    def prepro_instance(self):
        """Create PreProData instance."""
        return PreProData('/tmp/test_data')

    @pytest.mark.integration
    def test_nan_handling_integration(self, prepro_instance):
        """Test NaN handling with realistic physiological data."""
        # Create realistic sensor data with occasional NaNs
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='100ms')
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.randn(1000) * 0.1

        # Introduce some NaN values (simulating sensor dropouts)
        signal[100:105] = np.nan
        signal[500] = np.nan
        signal[900:905] = np.nan

        df = pd.DataFrame({'signal': signal}, index=timestamps)

        result = prepro_instance.handle_nans(df)

        # Verify no NaNs remain
        assert not result.isnull().any().any()

        # Verify interpolation preserves signal characteristics
        assert result['signal'].mean() == pytest.approx(signal[~np.isnan(signal)].mean(), abs=0.1)
        assert result['signal'].std() == pytest.approx(signal[~np.isnan(signal)].std(), abs=0.1)

    @pytest.mark.integration
    def test_trim_and_marker_adjustment_pipeline(self, prepro_instance):
        """Test combined trimming and marker adjustment workflow."""
        # Create sample data
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=100, freq='1s')
        df = pd.DataFrame({'value': np.random.randn(100)}, index=timestamps)

        # Create markers
        base_time = datetime(2024, 1, 1, 10, 0, 10)
        markers = [
            ('Start', base_time, 'task', timedelta(seconds=5)),
            ('End', base_time + timedelta(seconds=30), 'task', timedelta(seconds=5))
        ]

        # Adjust markers by +2 seconds
        time_shift = timedelta(seconds=2)
        adjusted_markers = prepro_instance.adjust_marker_timestamps(markers, time_shift)

        # Trim data based on adjusted markers
        start_time = adjusted_markers[0][1]
        end_time = adjusted_markers[1][1]
        trimmed_data = prepro_instance.trim_data_to_events(df, start_time, end_time)

        # Verify pipeline worked correctly
        assert not trimmed_data.empty
        assert trimmed_data.index[0] >= start_time
        assert trimmed_data.index[-1] <= end_time
        assert len(trimmed_data) > 0


class TestLoadMneRaw:
    """Test suite for _load_mne_raw method."""

    @pytest.fixture
    def prepro_instance(self, temp_dir):
        """Create a PreProData instance."""
        return PreProData(temp_dir)

    @pytest.mark.unit
    def test_load_mne_raw_basic(self, prepro_instance):
        """Test basic MNE Raw creation from DataFrame."""
        # Create sample DataFrame (must include R_AUX channel)
        n_samples = 1000
        df = pd.DataFrame({
            'CH1': np.random.randn(n_samples),
            'CH2': np.random.randn(n_samples),
            'CH3': np.random.randn(n_samples),
            'R_AUX': np.zeros(n_samples)  # Required channel that gets dropped
        })
        sfreq = 256.0

        # Load as MNE Raw
        raw = prepro_instance._load_mne_raw(df, sfreq)

        # Verify it's an MNE Raw object
        assert isinstance(raw, mne.io.RawArray)
        assert raw.info['sfreq'] == sfreq
        # R_AUX should be dropped, so only 3 channels remain
        assert len(raw.ch_names) == 3
        assert raw.ch_names == ['CH1', 'CH2', 'CH3']
        assert raw.get_data().shape == (3, n_samples)

    @pytest.mark.unit
    def test_load_mne_raw_handles_nans(self, prepro_instance):
        """Test that _load_mne_raw handles NaN values via handle_nans."""
        # Create DataFrame with NaNs (must include R_AUX)
        df = pd.DataFrame({
            'CH1': [1.0, np.nan, 3.0, 4.0, 5.0],
            'CH2': [10.0, 20.0, np.nan, 40.0, 50.0],
            'R_AUX': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        sfreq = 100.0

        # Load as MNE Raw (should handle NaNs)
        raw = prepro_instance._load_mne_raw(df, sfreq)

        # Verify no NaNs in the data
        data = raw.get_data()
        assert not np.isnan(data).any()
        # Should have 2 channels after dropping R_AUX
        assert raw.get_data().shape[0] == 2

    @pytest.mark.unit
    def test_load_mne_raw_preserves_channel_names(self, prepro_instance):
        """Test that channel names are preserved correctly."""
        df = pd.DataFrame({
            'EEG_AF7': np.random.randn(100),
            'EEG_AF8': np.random.randn(100),
            'EEG_TP9': np.random.randn(100),
            'EEG_TP10': np.random.randn(100),
            'R_AUX': np.zeros(100)
        })
        sfreq = 256.0

        raw = prepro_instance._load_mne_raw(df, sfreq)

        # R_AUX should be dropped
        assert raw.ch_names == ['EEG_AF7', 'EEG_AF8', 'EEG_TP9', 'EEG_TP10']


class TestAddEventMarkers:
    """Test suite for add_event_markers method."""

    @pytest.fixture
    def prepro_instance(self, temp_dir):
        """Create a PreProData instance."""
        return PreProData(temp_dir)

    @pytest.mark.unit
    def test_add_event_markers_to_mne_raw(self, prepro_instance):
        """Test adding event markers to MNE Raw object."""
        # Create MNE Raw data
        n_channels = 4
        n_samples = 1000
        sfreq = 256.0
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=['CH1', 'CH2', 'CH3', 'CH4'],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info)

        # Create markers
        eeg_start_time = datetime(2024, 1, 1, 10, 0, 0)
        markers = [
            ('Start', datetime(2024, 1, 1, 10, 0, 1), 'task', timedelta(seconds=5)),
            ('Middle', datetime(2024, 1, 1, 10, 0, 2), 'task', timedelta(seconds=5)),
            ('End', datetime(2024, 1, 1, 10, 0, 3), 'task', timedelta(seconds=5))
        ]

        # Add markers
        annotated_raw = prepro_instance.add_event_markers(raw, markers, eeg_start_time)

        # Verify annotations were added
        assert annotated_raw.annotations is not None
        assert len(annotated_raw.annotations) == 3
        assert annotated_raw.annotations.description[0] == 'Start'
        assert annotated_raw.annotations.description[1] == 'Middle'
        assert annotated_raw.annotations.description[2] == 'End'

        # Check onset times (should be 1.0, 2.0, 3.0 seconds)
        assert annotated_raw.annotations.onset[0] == pytest.approx(1.0, abs=0.01)
        assert annotated_raw.annotations.onset[1] == pytest.approx(2.0, abs=0.01)
        assert annotated_raw.annotations.onset[2] == pytest.approx(3.0, abs=0.01)

    @pytest.mark.unit
    def test_add_event_markers_to_dataframe(self, prepro_instance):
        """Test adding event markers to DataFrame."""
        # Create DataFrame with datetime index
        timestamps = pd.date_range('2024-01-01 10:00:00', periods=10, freq='1s')
        df = pd.DataFrame({'value': np.random.randn(10)}, index=timestamps)

        # Create markers (some matching timestamps)
        eeg_start_time = datetime(2024, 1, 1, 10, 0, 0)
        markers = [
            ('Event1', datetime(2024, 1, 1, 10, 0, 2), 'task', timedelta(seconds=1)),
            ('Event2', datetime(2024, 1, 1, 10, 0, 5), 'task', timedelta(seconds=1))
        ]

        # Add markers (returns same DataFrame since it only works with MNE Raw)
        result = prepro_instance.add_event_markers(df, markers, eeg_start_time)

        # For DataFrame, method just returns the data unchanged
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    @pytest.mark.unit
    def test_add_event_markers_filters_out_of_range(self, prepro_instance, capsys):
        """Test that markers outside data range are filtered out."""
        # Create short MNE Raw data (2 seconds at 100 Hz)
        n_channels = 2
        n_samples = 200
        sfreq = 100.0
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=['CH1', 'CH2'],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info)

        # Create markers (some outside range)
        eeg_start_time = datetime(2024, 1, 1, 10, 0, 0)
        markers = [
            ('InRange', datetime(2024, 1, 1, 10, 0, 1), 'task', timedelta(seconds=1)),
            ('OutOfRange', datetime(2024, 1, 1, 10, 0, 5), 'task', timedelta(seconds=1)),  # 5 seconds > 2 second duration
        ]

        # Add markers
        annotated_raw = prepro_instance.add_event_markers(raw, markers, eeg_start_time)

        # Only the in-range marker should be added
        assert len(annotated_raw.annotations) == 1
        assert annotated_raw.annotations.description[0] == 'InRange'

    @pytest.mark.integration
    def test_add_event_markers_integration_with_load_mne_raw(self, prepro_instance):
        """Test adding markers to MNE Raw created via _load_mne_raw."""
        # Create DataFrame and convert to MNE Raw (must include R_AUX)
        df = pd.DataFrame({
            'CH1': np.random.randn(256),
            'CH2': np.random.randn(256),
            'R_AUX': np.zeros(256)
        })
        sfreq = 256.0
        raw = prepro_instance._load_mne_raw(df, sfreq)

        # Add markers
        eeg_start_time = datetime(2024, 1, 1, 10, 0, 0)
        markers = [
            ('Task1', datetime(2024, 1, 1, 10, 0, 0, 500000), 'task', timedelta(milliseconds=500))
        ]

        annotated_raw = prepro_instance.add_event_markers(raw, markers, eeg_start_time)

        # Verify marker was added
        assert len(annotated_raw.annotations) == 1
        assert annotated_raw.annotations.description[0] == 'Task1'
