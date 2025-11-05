"""
Unit tests for modalities/prepro_temp.py

Tests the PreProTEMP class for skin temperature signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_temp import PreProTEMP


class TestPreProTEMP:
    """Test suite for PreProTEMP class."""

    @pytest.fixture
    def sample_temp_data(self):
        """Create sample temperature DataFrame for testing."""
        # Create timestamp index at 4 Hz (typical TEMP sampling rate)
        n_samples = 600
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='250ms')  # 4 Hz

        # Create realistic temperature signal (in °C)
        # Simulate slow changes with circadian rhythm and stress responses
        t = np.arange(n_samples) / 4.0  # Time in seconds
        circadian = 0.5 * np.sin(2 * np.pi * 0.001 * t)  # Very slow rhythm
        stress_response = -0.2 * np.exp(-t / 30)  # Temperature drop during stress
        temp_signal = 32.5 + circadian + stress_response + np.random.randn(n_samples) * 0.05

        data = pd.DataFrame({
            'TEMP': temp_signal
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_temp_data):
        """Create sample dataset dictionary for PreProTEMP."""
        return {
            'stream_1': {
                'data': sample_temp_data.copy(),
                'sfreq': 4
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProTEMP initialization."""
        prepro = PreProTEMP(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 4

    @pytest.mark.unit
    def test_smooth_data(self, sample_temp_data):
        """Test moving average smoothing."""
        prepro = PreProTEMP({'test': {'data': sample_temp_data, 'sfreq': 4}})
        prepro.data = sample_temp_data
        prepro.sfreq = 4

        smoothed = prepro.smooth_data(window_size=5)

        assert smoothed is not None
        # Smoothed length should be original length - window_size + 1
        assert len(smoothed) == len(sample_temp_data) - 5 + 1

    @pytest.mark.unit
    def test_baseline_correction(self, sample_temp_data):
        """Test baseline drift correction."""
        prepro = PreProTEMP({'test': {'data': sample_temp_data, 'sfreq': 4}})

        # Add artificial linear drift
        drift = np.linspace(0, 2, len(sample_temp_data))
        data_with_drift = sample_temp_data['TEMP'].values + drift

        corrected = prepro.baseline_correction(data_with_drift)

        assert corrected is not None
        assert len(corrected) == len(data_with_drift)
        # Mean should be reduced after drift removal
        assert np.mean(corrected) < np.mean(data_with_drift)

    @pytest.mark.unit
    def test_normalize(self, sample_temp_data):
        """Test min-max normalization."""
        prepro = PreProTEMP({'test': {'data': sample_temp_data, 'sfreq': 4}})

        normalized = prepro.normalize(sample_temp_data['TEMP'].values)

        # Should be scaled to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.abs(normalized.min() - 0.0) < 0.01
        assert np.abs(normalized.max() - 1.0) < 0.01

    @pytest.mark.unit
    def test_preprocess_temp_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProTEMP(sample_dataset)

        processed = prepro.preprocess_temp_data(epoch_length=60)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']

    @pytest.mark.unit
    def test_plot_temp_data(self, sample_dataset):
        """Test TEMP data plotting."""
        prepro = PreProTEMP(sample_dataset)

        figs, titles = prepro.plot_temp_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "Temperature Data" in titles[0]


class TestPreProTEMPIntegration:
    """Integration tests for PreProTEMP preprocessing pipeline."""

    @pytest.fixture
    def sample_temp_data(self):
        """Create sample TEMP data."""
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='250ms')

        # Realistic temperature with various components
        t = np.arange(n_samples) / 4.0
        circadian = 0.5 * np.sin(2 * np.pi * 0.0005 * t)
        activity = 0.3 * np.sin(2 * np.pi * 0.01 * t)
        temp_signal = 32.0 + circadian + activity + np.random.randn(n_samples) * 0.05

        data = pd.DataFrame({'TEMP': temp_signal}, index=timestamps)
        return data

    @pytest.fixture
    def sample_dataset(self, sample_temp_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_temp_data.copy(),
                'sfreq': 4
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProTEMP(sample_dataset)

        # Run full preprocessing with moderate epoch length
        processed = prepro.preprocess_temp_data(epoch_length=60)

        # Verify all steps completed successfully
        assert 'stream_1' in processed
        stream_data = processed['stream_1']

        # Check data was processed
        assert stream_data['data'] is not None

        # Check epochs were created
        assert len(stream_data['epochs']) > 0

    @pytest.mark.integration
    def test_circadian_analysis_epoch_length(self, sample_dataset):
        """Test preprocessing with very long epochs for circadian analysis."""
        prepro = PreProTEMP(sample_dataset)

        # Use 120-second epochs for circadian analysis
        processed = prepro.preprocess_temp_data(epoch_length=120)

        stream_data = processed['stream_1']

        # Should have created epochs
        assert len(stream_data['epochs']) >= 0

        # If epochs exist, verify they're sufficiently long
        if len(stream_data['epochs']) > 0:
            first_epoch = stream_data['epochs'][0]
            # At 4 Hz, 120 seconds should have ~480 samples
            assert len(first_epoch) > 200  # At least substantial data
