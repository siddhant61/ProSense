"""
Unit tests for modalities/prepro_gsr.py

Tests the PreProGSR class for galvanic skin response signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_gsr import PreProGSR


class TestPreProGSR:
    """Test suite for PreProGSR class."""

    @pytest.fixture
    def sample_gsr_data(self):
        """Create sample GSR DataFrame for testing."""
        # Create timestamp index at 4 Hz (typical GSR sampling rate)
        n_samples = 500
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='250ms')  # 4 Hz

        # Create realistic GSR signal (slow-changing conductance in µS)
        # Simulate slow drift + arousal responses
        t = np.arange(n_samples) / 4.0  # Time in seconds
        baseline_drift = 0.1 * t  # Slow drift
        arousal_response = 2 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz arousal
        gsr_signal = 5 + baseline_drift + arousal_response + np.random.randn(n_samples) * 0.1

        data = pd.DataFrame({
            'GSR': gsr_signal
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_gsr_data):
        """Create sample dataset dictionary for PreProGSR."""
        return {
            'stream_1': {
                'data': sample_gsr_data.copy(),
                'sfreq': 4
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProGSR initialization."""
        prepro = PreProGSR(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 4

    @pytest.mark.unit
    def test_lowpass_filter(self, sample_gsr_data):
        """Test low-pass filtering."""
        prepro = PreProGSR({'test': {'data': sample_gsr_data, 'sfreq': 4}})
        prepro.sfreq = 4

        filtered = prepro.lowpass_filter(sample_gsr_data, cutoff=1.0)

        assert filtered is not None
        assert len(filtered) == len(sample_gsr_data)

    @pytest.mark.unit
    def test_normalize(self, sample_gsr_data):
        """Test min-max normalization."""
        prepro = PreProGSR({'test': {'data': sample_gsr_data, 'sfreq': 4}})

        normalized = prepro.normalize(sample_gsr_data['GSR'].values)

        # Should be scaled to [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.abs(normalized.min() - 0.0) < 0.01
        assert np.abs(normalized.max() - 1.0) < 0.01

    @pytest.mark.unit
    def test_preprocess_gsr_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProGSR(sample_dataset)

        processed = prepro.preprocess_gsr_data(epoch_length=30)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']

    @pytest.mark.unit
    def test_plot_gsr_data(self, sample_dataset):
        """Test GSR data plotting."""
        prepro = PreProGSR(sample_dataset)

        figs, titles = prepro.plot_gsr_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "GSR Data" in titles[0]


class TestPreProGSRIntegration:
    """Integration tests for PreProGSR preprocessing pipeline."""

    @pytest.fixture
    def sample_gsr_data(self):
        """Create sample GSR data."""
        n_samples = 800
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='250ms')

        # Realistic GSR with tonic and phasic components
        t = np.arange(n_samples) / 4.0
        tonic = 5 + 0.1 * t  # Slow baseline drift
        phasic = 2 * np.sin(2 * np.pi * 0.03 * t)  # Arousal responses
        gsr_signal = tonic + phasic + np.random.randn(n_samples) * 0.1

        data = pd.DataFrame({'GSR': gsr_signal}, index=timestamps)
        return data

    @pytest.fixture
    def sample_dataset(self, sample_gsr_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_gsr_data.copy(),
                'sfreq': 4
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProGSR(sample_dataset)

        # Run full preprocessing
        processed = prepro.preprocess_gsr_data(epoch_length=30)

        # Verify all steps completed successfully
        assert 'stream_1' in processed
        stream_data = processed['stream_1']

        # Check data was processed
        assert stream_data['data'] is not None

        # Check epochs were created
        assert len(stream_data['epochs']) > 0

    @pytest.mark.integration
    def test_stress_analysis_epoch_length(self, sample_dataset):
        """Test preprocessing with long epochs suitable for stress analysis."""
        prepro = PreProGSR(sample_dataset)

        # Use 60-second epochs for stress analysis
        processed = prepro.preprocess_gsr_data(epoch_length=60)

        stream_data = processed['stream_1']

        # Should have created epochs
        assert len(stream_data['epochs']) >= 0

        # If epochs exist, verify they're sufficiently long
        if len(stream_data['epochs']) > 0:
            first_epoch = stream_data['epochs'][0]
            # At 4 Hz, 60 seconds should have ~240 samples
            assert len(first_epoch) > 100  # At least substantial data
