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
