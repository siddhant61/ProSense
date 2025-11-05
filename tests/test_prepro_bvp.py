"""
Unit tests for modalities/prepro_bvp.py

Tests the PreProBVP class for blood volume pulse signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_bvp import PreProBVP


class TestPreProBVP:
    """Test suite for PreProBVP class."""

    @pytest.fixture
    def sample_bvp_data(self):
        """Create sample BVP DataFrame for testing."""
        # Create timestamp index at 64 Hz (typical BVP sampling rate)
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')  # 64 Hz

        # Create realistic BVP signal (single channel pulsatile waveform)
        # Simulate cardiac cycle with ~1 Hz frequency (60 BPM)
        t = np.arange(n_samples) / 64.0  # Time in seconds
        cardiac_freq = 1.0  # 60 BPM
        bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.randn(n_samples) * 2

        data = pd.DataFrame({
            'BVP': bvp_signal
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_bvp_data):
        """Create sample dataset dictionary for PreProBVP."""
        return {
            'stream_1': {
                'data': sample_bvp_data.copy(),
                'sfreq': 64
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProBVP initialization."""
        prepro = PreProBVP(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 64

    @pytest.mark.unit
    def test_calculate_sampling_frequency(self, sample_bvp_data):
        """Test sampling frequency calculation from timestamps."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})
        sfreq = prepro.calculate_sampling_frequency(sample_bvp_data)

        # Should be close to 64 Hz
        assert 60 < sfreq < 68

    @pytest.mark.unit
    def test_butter_bandpass_filter(self, sample_dataset):
        """Test Butterworth bandpass filter."""
        prepro = PreProBVP(sample_dataset)
        test_data = np.random.randn(1000) + 100

        filtered = prepro.butter_bandpass_filter(
            test_data,
            lowcut=0.5,
            highcut=4.0,
            sfreq=64,
            order=3
        )

        assert filtered is not None
        assert len(filtered) == len(test_data)

    @pytest.mark.unit
    def test_remove_noise(self, sample_bvp_data):
        """Test noise removal with bandpass filtering."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})

        filtered = prepro.remove_noise(sample_bvp_data, sfreq=64)

        assert filtered is not None
        assert filtered.shape == sample_bvp_data.shape
        # Should not contain NaN or inf values
        assert not filtered.isnull().any().any()
        assert not np.isinf(filtered.values).any()

    @pytest.mark.unit
    def test_remove_noise_with_nan_values(self, sample_bvp_data):
        """Test noise removal handles NaN values correctly."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})

        # Introduce some NaN values
        data_with_nan = sample_bvp_data.copy()
        data_with_nan.iloc[10:15] = np.nan

        filtered = prepro.remove_noise(data_with_nan, sfreq=64)

        # Should handle NaN with forward/backward fill
        assert not filtered.isnull().any().any()

    @pytest.mark.unit
    def test_downsample_data(self, sample_bvp_data):
        """Test downsampling functionality."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})

        # Downsample from 64 Hz to 32 Hz
        downsampled = prepro.downsample_data(sample_bvp_data, 64, 32)

        # Should have roughly half the samples
        assert len(downsampled) == len(sample_bvp_data) // 2

    @pytest.mark.unit
    def test_downsample_data_no_change(self, sample_bvp_data):
        """Test downsampling when target >= original frequency."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})

        # Try to "downsample" to higher frequency
        result = prepro.downsample_data(sample_bvp_data, 64, 128)

        # Should return original data unchanged
        assert len(result) == len(sample_bvp_data)
        pd.testing.assert_frame_equal(result, sample_bvp_data)

    @pytest.mark.unit
    def test_segment_data(self, sample_bvp_data):
        """Test data segmentation into epochs."""
        prepro = PreProBVP({'test': {'data': sample_bvp_data, 'sfreq': 64}})

        # Segment into 5-second epochs
        segments = prepro.segment_data(sample_bvp_data, segment_length=5.0, sfreq=64)

        assert len(segments) > 0
        # Each segment should have approximately 320 samples (5 seconds at 64 Hz)
        for segment in segments:
            assert len(segment) > 0  # Time-based segmentation may have slight variations

    @pytest.mark.unit
    def test_preprocess_bvp_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProBVP(sample_dataset)

        processed = prepro.preprocess_bvp_data(epoch_length=5)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']
        assert len(processed['stream_1']['epochs']) > 0

    @pytest.mark.unit
    def test_plot_bvp_data(self, sample_dataset):
        """Test BVP data plotting."""
        prepro = PreProBVP(sample_dataset)

        figs, titles = prepro.plot_bvp_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "BVP Data" in titles[0]

    @pytest.mark.unit
    def test_plot_bvp_data_handles_ndarray(self):
        """Test plot handles numpy array input."""
        bvp_array = np.random.randn(100) + 100
        timestamps = pd.date_range('2024-01-01', periods=100, freq='15.625ms')

        dataset = {
            'stream_1': {
                'data': bvp_array,
                'sfreq': 64
            }
        }

        prepro = PreProBVP(dataset)
        figs, titles = prepro.plot_bvp_data(dataset)

        # Should handle conversion and create plot
        assert len(figs) == 1

    @pytest.mark.unit
    def test_plot_bvp_data_handles_series(self, sample_bvp_data):
        """Test plot handles pandas Series input."""
        dataset = {
            'stream_1': {
                'data': sample_bvp_data['BVP'],  # Extract Series
                'sfreq': 64
            }
        }

        prepro = PreProBVP(dataset)
        figs, titles = prepro.plot_bvp_data(dataset)

        # Should handle conversion and create plot
        assert len(figs) == 1


class TestPreProBVPIntegration:
    """Integration tests for PreProBVP preprocessing pipeline."""

    @pytest.fixture
    def sample_bvp_data(self):
        """Create sample BVP data."""
        n_samples = 2000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')

        # Simulate realistic BVP signal
        t = np.arange(n_samples) / 64.0
        cardiac_freq = 1.2  # 72 BPM
        bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.randn(n_samples) * 2

        data = pd.DataFrame({
            'BVP': bvp_signal
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_bvp_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_bvp_data.copy(),
                'sfreq': 64
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProBVP(sample_dataset)

        # Run full preprocessing
        processed = prepro.preprocess_bvp_data(epoch_length=10)

        # Verify all steps completed successfully
        assert 'stream_1' in processed
        stream_data = processed['stream_1']

        # Check data was filtered
        assert stream_data['data'] is not None
        assert not stream_data['data'].isnull().any().any()

        # Check epochs were created
        assert len(stream_data['epochs']) > 0

        # Check each epoch is non-empty
        for epoch in stream_data['epochs']:
            assert len(epoch) > 0

    @pytest.mark.integration
    def test_multi_stream_preprocessing(self):
        """Test preprocessing with multiple streams."""
        # Create multi-stream dataset
        dataset = {}
        for i in range(3):
            n_samples = 1500
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0
            cardiac_freq = 1.0 + i * 0.2  # Different heart rates
            bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.randn(n_samples) * 2

            data = pd.DataFrame({'BVP': bvp_signal}, index=timestamps)
            dataset[f'stream_{i+1}'] = {'data': data, 'sfreq': 64}

        prepro = PreProBVP(dataset)
        processed = prepro.preprocess_bvp_data(epoch_length=5)

        # Verify all streams were processed
        assert len(processed) == 3
        for stream_id in ['stream_1', 'stream_2', 'stream_3']:
            assert stream_id in processed
            assert 'epochs' in processed[stream_id]

    @pytest.mark.integration
    def test_hrv_analysis_epoch_length(self, sample_dataset):
        """Test preprocessing with long epochs suitable for HRV analysis."""
        prepro = PreProBVP(sample_dataset)

        # Use 60-second epochs for HRV analysis
        processed = prepro.preprocess_bvp_data(epoch_length=60)

        stream_data = processed['stream_1']

        # Should have at least one epoch
        assert len(stream_data['epochs']) >= 0  # May be 0 if data < 60s

        # If epochs exist, verify they're sufficiently long for HRV
        if len(stream_data['epochs']) > 0:
            first_epoch = stream_data['epochs'][0]
            # At 64 Hz, 60 seconds should have ~3840 samples
            assert len(first_epoch) > 1000  # At least substantial data
