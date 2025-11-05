"""
Unit tests for modalities/prepro_gyro.py

Tests the PreProGYRO class for gyroscope signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_gyro import PreProGYRO


class TestPreProGYRO:
    """Test suite for PreProGYRO class."""

    @pytest.fixture
    def sample_gyro_data(self):
        """Create sample 3-axis gyroscope DataFrame for testing."""
        # Create timestamp index at 32 Hz (typical GYRO sampling rate)
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')  # 32 Hz

        # Create realistic GYRO signal (3 axes: X, Y, Z)
        # Angular velocity typically in range of -250 to +250 degrees/second
        data = pd.DataFrame({
            'X': np.random.randn(n_samples) * 50 + 10,   # X-axis rotation
            'Y': np.random.randn(n_samples) * 50 - 5,    # Y-axis rotation
            'Z': np.random.randn(n_samples) * 50 + 2     # Z-axis rotation
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_gyro_data):
        """Create sample dataset dictionary for PreProGYRO."""
        return {
            'stream_1': {
                'data': sample_gyro_data.copy(),
                'sfreq': 32
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProGYRO initialization."""
        prepro = PreProGYRO(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 32

    @pytest.mark.unit
    def test_calculate_sampling_frequency(self, sample_gyro_data):
        """Test sampling frequency calculation from timestamps."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})
        sfreq = prepro.calculate_sampling_frequency(sample_gyro_data)

        # Should be close to 32 Hz
        assert 30 < sfreq < 34

    @pytest.mark.unit
    def test_butter_lowpass_filter(self, sample_dataset):
        """Test Butterworth low-pass filter."""
        prepro = PreProGYRO(sample_dataset)
        test_data = np.random.randn(1000)

        filtered = prepro.butter_lowpass_filter(
            test_data,
            cutoff=5.0,
            sfreq=32,
            order=5
        )

        assert filtered is not None
        assert len(filtered) == len(test_data)

    @pytest.mark.unit
    def test_remove_noise(self, sample_gyro_data):
        """Test noise removal with low-pass filtering."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})

        filtered = prepro.remove_noise(sample_gyro_data, sfreq=32)

        assert filtered is not None
        assert filtered.shape == sample_gyro_data.shape
        # Should not contain NaN or inf values
        assert not filtered.isnull().any().any()
        assert not np.isinf(filtered.values).any()

    @pytest.mark.unit
    def test_remove_noise_with_nan_values(self, sample_gyro_data):
        """Test noise removal handles NaN values correctly."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})

        # Introduce some NaN values
        data_with_nan = sample_gyro_data.copy()
        data_with_nan.iloc[10:15] = np.nan

        filtered = prepro.remove_noise(data_with_nan, sfreq=32)

        # Should handle NaN with forward/backward fill
        assert not filtered.isnull().any().any()

    @pytest.mark.unit
    def test_normalize_data(self, sample_gyro_data):
        """Test z-score normalization."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})
        normalized = prepro.normalize_data(sample_gyro_data)

        # Check that data is normalized (approximately zero mean, unit variance)
        assert np.abs(normalized.mean().mean()) < 0.1  # Close to 0
        assert np.abs(normalized.std().mean() - 1.0) < 0.1  # Close to 1

    @pytest.mark.unit
    def test_infer_sensor_location_muse(self, sample_dataset):
        """Test sensor location inference for Muse (head) sensor."""
        prepro = PreProGYRO(sample_dataset)
        location = prepro.infer_sensor_location('muse_gyro_stream')
        assert location == 'head'

    @pytest.mark.unit
    def test_infer_sensor_location_wrist(self, sample_dataset):
        """Test sensor location inference for wrist sensor."""
        prepro = PreProGYRO(sample_dataset)
        location = prepro.infer_sensor_location('empatica_1')
        assert location == 'wrist'

    @pytest.mark.unit
    def test_infer_sensor_location_unknown(self, sample_dataset):
        """Test sensor location inference for unknown sensor."""
        prepro = PreProGYRO(sample_dataset)
        location = prepro.infer_sensor_location('unknown_sensor')
        assert location == 'unknown'

    @pytest.mark.unit
    def test_downsample_data(self, sample_gyro_data):
        """Test downsampling functionality."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})

        # Downsample from 32 Hz to 16 Hz
        downsampled = prepro.downsample_data(sample_gyro_data, 32, 16)

        # Should have roughly half the samples
        assert len(downsampled) == len(sample_gyro_data) // 2

    @pytest.mark.unit
    def test_segment_data(self, sample_gyro_data):
        """Test data segmentation into epochs."""
        prepro = PreProGYRO({'test': {'data': sample_gyro_data, 'sfreq': 32}})

        # Segment into 1-second epochs
        segments = prepro.segment_data(sample_gyro_data, segment_length=1.0, sfreq=32)

        assert len(segments) > 0
        # Each segment should have 32 samples (1 second at 32 Hz)
        for segment in segments:
            assert len(segment) == 32

    @pytest.mark.unit
    def test_preprocess_gyro_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProGYRO(sample_dataset)

        processed = prepro.preprocess_gyro_data(epoch_length=5)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']
        assert len(processed['stream_1']['epochs']) > 0

    @pytest.mark.unit
    def test_plot_gyro_data(self, sample_dataset):
        """Test GYRO data plotting."""
        prepro = PreProGYRO(sample_dataset)

        figs, titles = prepro.plot_gyro_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "Gyroscope Data" in titles[0]


class TestPreProGYROIntegration:
    """Integration tests for PreProGYRO preprocessing pipeline."""

    @pytest.fixture
    def sample_gyro_data(self):
        """Create sample GYRO data."""
        n_samples = 2000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')

        data = pd.DataFrame({
            'X': np.random.randn(n_samples) * 50 + 10,
            'Y': np.random.randn(n_samples) * 50 - 5,
            'Z': np.random.randn(n_samples) * 50 + 2
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_gyro_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_gyro_data.copy(),
                'sfreq': 32
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProGYRO(sample_dataset)

        # Run full preprocessing
        processed = prepro.preprocess_gyro_data(epoch_length=10)

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
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')
            data = pd.DataFrame({
                'X': np.random.randn(n_samples) * 50,
                'Y': np.random.randn(n_samples) * 50,
                'Z': np.random.randn(n_samples) * 50
            }, index=timestamps)
            dataset[f'stream_{i+1}'] = {'data': data, 'sfreq': 32}

        prepro = PreProGYRO(dataset)
        processed = prepro.preprocess_gyro_data(epoch_length=5)

        # Verify all streams were processed
        assert len(processed) == 3
        for stream_id in ['stream_1', 'stream_2', 'stream_3']:
            assert stream_id in processed
            assert 'epochs' in processed[stream_id]
