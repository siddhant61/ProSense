"""
Unit tests for modalities/prepro_ppg.py

Tests the PreProPPG class for PPG signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_ppg import PreProPPG


class TestPreProPPG:
    """Test suite for PreProPPG class."""

    @pytest.fixture
    def sample_ppg_data(self):
        """Create sample PPG DataFrame for testing."""
        # Create timestamp index at 64 Hz (typical PPG sampling rate)
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')  # 64 Hz

        # Create realistic PPG signal (3 channels: Ambient, IR, Red)
        data = pd.DataFrame({
            'Ambient': np.random.randn(n_samples) * 10 + 100,
            'IR': np.random.randn(n_samples) * 50 + 500,
            'Red': np.random.randn(n_samples) * 30 + 300
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_ppg_data):
        """Create sample dataset dictionary for PreProPPG."""
        return {
            'stream_1': {
                'data': sample_ppg_data.copy(),
                'sfreq': 64
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProPPG initialization."""
        prepro = PreProPPG(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 64

    @pytest.mark.unit
    def test_calculate_sampling_frequency(self, sample_ppg_data):
        """Test sampling frequency calculation from timestamps."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})
        sfreq = prepro.calculate_sampling_frequency(sample_ppg_data)

        # Should be close to 64 Hz
        assert 60 < sfreq < 68

    @pytest.mark.unit
    def test_calculate_sampling_frequency_invalid_index(self, sample_dataset):
        """Test sampling frequency calculation with non-datetime index."""
        prepro = PreProPPG(sample_dataset)

        # Create DataFrame without datetime index
        data = pd.DataFrame({'value': np.random.randn(100)})

        with pytest.raises(ValueError, match="must be a DatetimeIndex"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_calculate_sampling_frequency_insufficient_data(self, sample_dataset):
        """Test sampling frequency calculation with insufficient data."""
        prepro = PreProPPG(sample_dataset)

        # Create DataFrame with only 1 sample
        timestamps = pd.date_range('2024-01-01', periods=1, freq='1s')
        data = pd.DataFrame({'value': [1.0]}, index=timestamps)

        with pytest.raises(ValueError, match="not have enough data points"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_butter_bandpass_filter(self, sample_dataset):
        """Test Butterworth bandpass filter."""
        prepro = PreProPPG(sample_dataset)
        test_data = np.random.randn(1000)

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
    def test_normalize_data(self, sample_ppg_data):
        """Test z-score normalization."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})
        normalized = prepro.normalize_data(sample_ppg_data)

        # Check that data is normalized (approximately zero mean, unit variance)
        assert np.abs(normalized.mean().mean()) < 0.1  # Close to 0
        assert np.abs(normalized.std().mean() - 1.0) < 0.1  # Close to 1

    @pytest.mark.unit
    def test_downsample_data(self, sample_ppg_data):
        """Test downsampling functionality."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        # Downsample from 64 Hz to 32 Hz
        downsampled = prepro.downsample_data(sample_ppg_data, 64, 32)

        # Should have roughly half the samples
        assert len(downsampled) == len(sample_ppg_data) // 2

    @pytest.mark.unit
    def test_downsample_data_no_change(self, sample_ppg_data):
        """Test downsampling when target >= original frequency."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        # Try to "downsample" to higher frequency
        result = prepro.downsample_data(sample_ppg_data, 64, 128)

        # Should return original data unchanged
        assert len(result) == len(sample_ppg_data)
        pd.testing.assert_frame_equal(result, sample_ppg_data)

    @pytest.mark.unit
    def test_segment_data(self, sample_ppg_data):
        """Test data segmentation into epochs."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        # Segment into 1-second epochs
        segments = prepro.segment_data(sample_ppg_data, segment_length=1.0, sfreq=64)

        assert len(segments) > 0
        # Each segment should have 64 samples (1 second at 64 Hz)
        for segment in segments:
            assert len(segment) == 64

    @pytest.mark.unit
    def test_segment_data_partial_segments_discarded(self, sample_ppg_data):
        """Test that partial segments are discarded."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        # Use a segment length that doesn't divide evenly
        segments = prepro.segment_data(sample_ppg_data, segment_length=7.0, sfreq=64)

        # Each segment should be exactly 7 seconds (448 samples)
        for segment in segments:
            assert len(segment) == 448

    @pytest.mark.unit
    def test_remove_noise(self, sample_ppg_data):
        """Test noise removal with bandpass filtering."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        filtered = prepro.remove_noise(sample_ppg_data, sfreq=64)

        assert filtered is not None
        assert filtered.shape == sample_ppg_data.shape
        # Should not contain NaN or inf values
        assert not filtered.isnull().any().any()
        assert not np.isinf(filtered.values).any()

    @pytest.mark.unit
    def test_remove_noise_with_nan_values(self, sample_ppg_data):
        """Test noise removal handles NaN values correctly."""
        prepro = PreProPPG({'test': {'data': sample_ppg_data, 'sfreq': 64}})

        # Introduce some NaN values
        data_with_nan = sample_ppg_data.copy()
        data_with_nan.iloc[10:15] = np.nan

        filtered = prepro.remove_noise(data_with_nan, sfreq=64)

        # Should handle NaN with forward/backward fill
        assert not filtered.isnull().any().any()

    @pytest.mark.unit
    def test_preprocess_ppg_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProPPG(sample_dataset)

        processed = prepro.preprocess_ppg_data(epoch_length=5)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']
        assert len(processed['stream_1']['epochs']) > 0

    @pytest.mark.unit
    def test_plot_ppg_data(self, sample_dataset):
        """Test PPG data plotting."""
        prepro = PreProPPG(sample_dataset)

        figs, titles = prepro.plot_ppg_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "PPG Data" in titles[0]


class TestPreProPPGIntegration:
    """Integration tests for PreProPPG preprocessing pipeline."""

    @pytest.fixture
    def sample_ppg_data(self):
        """Create sample PPG data."""
        n_samples = 2000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')

        data = pd.DataFrame({
            'Ambient': np.random.randn(n_samples) * 10 + 100,
            'IR': np.random.randn(n_samples) * 50 + 500,
            'Red': np.random.randn(n_samples) * 30 + 300
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_ppg_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_ppg_data.copy(),
                'sfreq': 64
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProPPG(sample_dataset)

        # Run full preprocessing
        processed = prepro.preprocess_ppg_data(epoch_length=10)

        # Verify all steps completed successfully
        assert 'stream_1' in processed
        stream_data = processed['stream_1']

        # Check data is normalized
        assert np.abs(stream_data['data'].mean().mean()) < 0.5
        assert 0.5 < stream_data['data'].std().mean() < 1.5

        # Check epochs were created
        assert len(stream_data['epochs']) > 0

        # Check each epoch has correct length
        expected_samples = int(10 * stream_data['sfreq'])
        for epoch in stream_data['epochs']:
            assert len(epoch) == expected_samples

    @pytest.mark.integration
    def test_multi_stream_preprocessing(self):
        """Test preprocessing with multiple streams."""
        # Create multi-stream dataset
        dataset = {}
        for i in range(3):
            n_samples = 1500
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            data = pd.DataFrame({
                'Ambient': np.random.randn(n_samples) * 10 + 100,
                'IR': np.random.randn(n_samples) * 50 + 500,
                'Red': np.random.randn(n_samples) * 30 + 300
            }, index=timestamps)
            dataset[f'stream_{i+1}'] = {'data': data, 'sfreq': 64}

        prepro = PreProPPG(dataset)
        processed = prepro.preprocess_ppg_data(epoch_length=5)

        # Verify all streams were processed
        assert len(processed) == 3
        for stream_id in ['stream_1', 'stream_2', 'stream_3']:
            assert stream_id in processed
            assert 'epochs' in processed[stream_id]

    @pytest.mark.integration
    def test_different_epoch_lengths(self, sample_dataset):
        """Test preprocessing with different epoch lengths."""
        prepro = PreProPPG(sample_dataset)

        for epoch_length in [1, 5, 10]:
            processed = prepro.preprocess_ppg_data(epoch_length=epoch_length)

            stream_data = processed['stream_1']
            expected_samples = int(epoch_length * stream_data['sfreq'])

            # Verify epoch length
            if len(stream_data['epochs']) > 0:
                assert len(stream_data['epochs'][0]) == expected_samples
