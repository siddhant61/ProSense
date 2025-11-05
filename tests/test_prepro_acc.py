"""
Unit tests for modalities/prepro_acc.py

Tests the PreProACC class for accelerometer signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.prepro_acc import PreProACC


class TestPreProACC:
    """Test suite for PreProACC class."""

    @pytest.fixture
    def sample_acc_data(self):
        """Create sample 3-axis accelerometer DataFrame for testing."""
        # Create timestamp index at 32 Hz (typical ACC sampling rate)
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')  # 32 Hz

        # Create realistic ACC signal (3 axes: X, Y, Z)
        # Accelerometer values typically in range of -2g to +2g (≈ -20 to +20 m/s²)
        data = pd.DataFrame({
            'X': np.random.randn(n_samples) * 2 + 0.5,  # X-axis with slight offset
            'Y': np.random.randn(n_samples) * 2 - 0.3,  # Y-axis
            'Z': np.random.randn(n_samples) * 2 + 9.8   # Z-axis (gravity component)
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_acc_data):
        """Create sample dataset dictionary for PreProACC."""
        return {
            'stream_1': {
                'data': sample_acc_data.copy(),
                'sfreq': 32
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProACC initialization."""
        prepro = PreProACC(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 32

    @pytest.mark.unit
    def test_calculate_sampling_frequency(self, sample_acc_data):
        """Test sampling frequency calculation from timestamps."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})
        sfreq = prepro.calculate_sampling_frequency(sample_acc_data)

        # Should be close to 32 Hz
        assert 30 < sfreq < 34

    @pytest.mark.unit
    def test_calculate_sampling_frequency_invalid_index(self, sample_dataset):
        """Test sampling frequency calculation with non-datetime index."""
        prepro = PreProACC(sample_dataset)

        # Create DataFrame without datetime index
        data = pd.DataFrame({'value': np.random.randn(100)})

        with pytest.raises(ValueError, match="must be a DatetimeIndex"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_calculate_sampling_frequency_insufficient_data(self, sample_dataset):
        """Test sampling frequency calculation with insufficient data."""
        prepro = PreProACC(sample_dataset)

        # Create DataFrame with only 1 sample
        timestamps = pd.date_range('2024-01-01', periods=1, freq='1s')
        data = pd.DataFrame({'value': [1.0]}, index=timestamps)

        with pytest.raises(ValueError, match="not have enough data points"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_butter_lowpass_filter(self, sample_dataset):
        """Test Butterworth low-pass filter."""
        prepro = PreProACC(sample_dataset)
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
    def test_remove_noise(self, sample_acc_data):
        """Test noise removal with low-pass filtering."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        filtered = prepro.remove_noise(sample_acc_data, sfreq=32)

        assert filtered is not None
        assert filtered.shape == sample_acc_data.shape
        # Should not contain NaN or inf values
        assert not filtered.isnull().any().any()
        assert not np.isinf(filtered.values).any()

    @pytest.mark.unit
    def test_remove_noise_with_nan_values(self, sample_acc_data):
        """Test noise removal handles NaN values correctly."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Introduce some NaN values
        data_with_nan = sample_acc_data.copy()
        data_with_nan.iloc[10:15] = np.nan

        filtered = prepro.remove_noise(data_with_nan, sfreq=32)

        # Should handle NaN with forward/backward fill
        assert not filtered.isnull().any().any()

    @pytest.mark.unit
    def test_handle_missing_values(self, sample_acc_data):
        """Test missing value handling with interpolation."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Introduce missing values
        data_with_missing = sample_acc_data.copy()
        data_with_missing.iloc[50:55] = np.nan

        interpolated = prepro.handle_missing_values(data_with_missing)

        # Should have no missing values after interpolation
        assert not interpolated.isnull().any().any()

    @pytest.mark.unit
    def test_normalize_data(self, sample_acc_data):
        """Test z-score normalization."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})
        normalized = prepro.normalize_data(sample_acc_data)

        # Check that data is normalized (approximately zero mean, unit variance)
        assert np.abs(normalized.mean().mean()) < 0.1  # Close to 0
        assert np.abs(normalized.std().mean() - 1.0) < 0.1  # Close to 1

    @pytest.mark.unit
    def test_infer_sensor_location_muse(self, sample_dataset):
        """Test sensor location inference for Muse (head) sensor."""
        prepro = PreProACC(sample_dataset)
        location = prepro.infer_sensor_location('muse_acc_stream')
        assert location == 'head'

    @pytest.mark.unit
    def test_infer_sensor_location_wrist(self, sample_dataset):
        """Test sensor location inference for wrist sensor."""
        prepro = PreProACC(sample_dataset)
        location = prepro.infer_sensor_location('empatica_1')
        assert location == 'wrist'

    @pytest.mark.unit
    def test_infer_sensor_location_unknown(self, sample_dataset):
        """Test sensor location inference for unknown sensor."""
        prepro = PreProACC(sample_dataset)
        location = prepro.infer_sensor_location('unknown_sensor')
        assert location == 'unknown'

    @pytest.mark.unit
    def test_downsample_data(self, sample_acc_data):
        """Test downsampling functionality."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Downsample from 32 Hz to 16 Hz
        downsampled = prepro.downsample_data(sample_acc_data, 32, 16)

        # Should have roughly half the samples
        assert len(downsampled) == len(sample_acc_data) // 2

    @pytest.mark.unit
    def test_downsample_data_no_change(self, sample_acc_data):
        """Test downsampling when target >= original frequency."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Try to "downsample" to higher frequency
        result = prepro.downsample_data(sample_acc_data, 32, 64)

        # Should return original data unchanged
        assert len(result) == len(sample_acc_data)
        pd.testing.assert_frame_equal(result, sample_acc_data)

    @pytest.mark.unit
    def test_segment_data(self, sample_acc_data):
        """Test data segmentation into epochs."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Segment into 1-second epochs
        segments = prepro.segment_data(sample_acc_data, epoch_length=1.0, sfreq=32)

        assert len(segments) > 0
        # Each segment should have 32 samples (1 second at 32 Hz)
        for segment in segments:
            assert len(segment) == 32

    @pytest.mark.unit
    def test_segment_data_partial_segments_discarded(self, sample_acc_data):
        """Test that partial segments are discarded."""
        prepro = PreProACC({'test': {'data': sample_acc_data, 'sfreq': 32}})

        # Use a segment length that doesn't divide evenly
        segments = prepro.segment_data(sample_acc_data, epoch_length=7.0, sfreq=32)

        # Each segment should be exactly 7 seconds (224 samples)
        for segment in segments:
            assert len(segment) == 224

    @pytest.mark.unit
    def test_preprocess_acc_data(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProACC(sample_dataset)

        processed = prepro.preprocess_acc_data(epoch_length=5)

        assert 'stream_1' in processed
        assert 'data' in processed['stream_1']
        assert 'epochs' in processed['stream_1']
        assert 'sfreq' in processed['stream_1']
        assert len(processed['stream_1']['epochs']) > 0

    @pytest.mark.unit
    def test_plot_acc_data(self, sample_dataset):
        """Test ACC data plotting."""
        prepro = PreProACC(sample_dataset)

        figs, titles = prepro.plot_acc_data(sample_dataset)

        assert len(figs) == 1
        assert len(titles) == 1
        assert "Accelerometer Data" in titles[0]


class TestPreProACCIntegration:
    """Integration tests for PreProACC preprocessing pipeline."""

    @pytest.fixture
    def sample_acc_data(self):
        """Create sample ACC data."""
        n_samples = 2000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')

        data = pd.DataFrame({
            'X': np.random.randn(n_samples) * 2 + 0.5,
            'Y': np.random.randn(n_samples) * 2 - 0.3,
            'Z': np.random.randn(n_samples) * 2 + 9.8
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_dataset(self, sample_acc_data):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_acc_data.copy(),
                'sfreq': 32
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline with all steps."""
        prepro = PreProACC(sample_dataset)

        # Run full preprocessing
        processed = prepro.preprocess_acc_data(epoch_length=10)

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
                'X': np.random.randn(n_samples) * 2 + 0.5,
                'Y': np.random.randn(n_samples) * 2 - 0.3,
                'Z': np.random.randn(n_samples) * 2 + 9.8
            }, index=timestamps)
            dataset[f'stream_{i+1}'] = {'data': data, 'sfreq': 32}

        prepro = PreProACC(dataset)
        processed = prepro.preprocess_acc_data(epoch_length=5)

        # Verify all streams were processed
        assert len(processed) == 3
        for stream_id in ['stream_1', 'stream_2', 'stream_3']:
            assert stream_id in processed
            assert 'epochs' in processed[stream_id]

    @pytest.mark.integration
    def test_different_epoch_lengths(self, sample_dataset):
        """Test preprocessing with different epoch lengths."""
        prepro = PreProACC(sample_dataset)

        for epoch_length in [1, 5, 10]:
            processed = prepro.preprocess_acc_data(epoch_length=epoch_length)

            stream_data = processed['stream_1']
            expected_samples = int(epoch_length * stream_data['sfreq'])

            # Verify epoch length
            if len(stream_data['epochs']) > 0:
                assert len(stream_data['epochs'][0]) == expected_samples

    @pytest.mark.integration
    def test_sensor_location_integration(self):
        """Test sensor location inference in full pipeline."""
        # Create datasets with different sensor locations
        n_samples = 1000
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')
        data = pd.DataFrame({
            'X': np.random.randn(n_samples) * 2,
            'Y': np.random.randn(n_samples) * 2,
            'Z': np.random.randn(n_samples) * 2 + 9.8
        }, index=timestamps)

        dataset = {
            'muse_acc': {'data': data.copy(), 'sfreq': 32},
            'empatica_1': {'data': data.copy(), 'sfreq': 32},
            'unknown_sensor': {'data': data.copy(), 'sfreq': 32}
        }

        prepro = PreProACC(dataset)

        # Test location inference
        assert prepro.infer_sensor_location('muse_acc') == 'head'
        assert prepro.infer_sensor_location('empatica_1') == 'wrist'
        assert prepro.infer_sensor_location('unknown_sensor') == 'unknown'

        # Verify preprocessing works with all sensor types
        processed = prepro.preprocess_acc_data(epoch_length=5)
        assert len(processed) == 3
