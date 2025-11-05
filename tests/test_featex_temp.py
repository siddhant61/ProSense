"""
Unit tests for modalities/featex_temp.py

Tests the FeatExTEMP class for temperature feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_temp import FeatExTEMP


class TestFeatExTEMP:
    """Test suite for FeatExTEMP class."""

    @pytest.fixture
    def sample_temp_epochs(self):
        """Create sample temperature epochs for testing."""
        # Create 3 epochs of temperature data
        epochs = []
        for i in range(3):
            timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')
            # Simulate temperature with slight variations
            temp_data = 32.5 + np.random.randn(100) * 0.5 + i * 0.2
            epoch = pd.DataFrame({'TEMP': temp_data}, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_temp_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_temp_epochs,
                'sfreq': 4
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExTEMP initialization."""
        featex = FeatExTEMP(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_compute_mean_temperature(self):
        """Test mean temperature computation."""
        featex = FeatExTEMP({})
        data = np.array([32.0, 32.5, 33.0, 32.5, 32.0])
        mean_temp = featex.compute_mean_temperature(data)

        assert isinstance(mean_temp, (int, float, np.number))
        assert 32.0 <= mean_temp <= 33.0

    @pytest.mark.unit
    def test_compute_max_temperature(self):
        """Test maximum temperature computation."""
        featex = FeatExTEMP({})
        data = np.array([32.0, 32.5, 33.5, 32.5, 32.0])
        max_temp = featex.compute_max_temperature(data)

        assert max_temp == 33.5

    @pytest.mark.unit
    def test_compute_min_temperature(self):
        """Test minimum temperature computation."""
        featex = FeatExTEMP({})
        data = np.array([32.0, 32.5, 33.0, 31.5, 32.0])
        min_temp = featex.compute_min_temperature(data)

        assert min_temp == 31.5

    @pytest.mark.unit
    def test_compute_temperature_variability(self):
        """Test temperature variability (std) computation."""
        featex = FeatExTEMP({})
        data = np.array([32.0, 32.5, 33.0, 32.5, 32.0])
        variability = featex.compute_temperature_variability(data)

        assert isinstance(variability, (int, float, np.number))
        assert variability >= 0  # Std is always non-negative

    @pytest.mark.unit
    def test_compute_rate_of_change(self):
        """Test rate of change computation."""
        featex = FeatExTEMP({})

        # Test with changing data
        data = np.array([32.0, 32.5, 33.0, 32.5, 32.0])
        rate = featex.compute_rate_of_change(data)
        assert isinstance(rate, (int, float, np.number))
        assert rate >= 0  # Mean absolute derivative is non-negative

        # Test with single sample (edge case)
        single_sample = np.array([32.0])
        rate_single = featex.compute_rate_of_change(single_sample)
        assert rate_single == 0

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExTEMP(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check we have features for all 3 epochs
        stream_features = features['stream_1']
        assert len(stream_features) == 3

        # Check first epoch has all required features
        first_epoch_features = stream_features[0]
        assert 'Min Temperature' in first_epoch_features
        assert 'Mean Temperature' in first_epoch_features
        assert 'Max Temperature' in first_epoch_features
        assert 'Temperature Variability' in first_epoch_features
        assert 'Rate of Change' in first_epoch_features

    @pytest.mark.unit
    def test_plot_features_over_epoch(self, sample_dataset):
        """Test plotting features over epochs."""
        featex = FeatExTEMP(sample_dataset)
        features = featex.extract_features()

        figs, titles = featex.plot_features_over_epoch(features)

        assert len(figs) == 1
        assert len(titles) == 1
        assert 'Temperature Features Over Epochs' in titles[0]


class TestFeatExTEMPIntegration:
    """Integration tests for FeatExTEMP feature extraction."""

    @pytest.fixture
    def realistic_temp_data(self):
        """Create realistic temperature data with circadian pattern."""
        epochs = []
        for i in range(5):
            timestamps = pd.date_range('2024-01-01', periods=200, freq='250ms')

            # Simulate temperature with circadian rhythm and noise
            t = np.arange(200) / 4.0  # Time in seconds
            circadian = 0.3 * np.sin(2 * np.pi * 0.001 * t)  # Very slow oscillation
            noise = np.random.randn(200) * 0.1
            temp_data = 32.0 + circadian + noise + i * 0.1  # Slight drift across epochs

            epoch = pd.DataFrame({'TEMP': temp_data}, index=timestamps)
            epochs.append(epoch)

        return {
            'stream_1': {
                'epochs': epochs,
                'sfreq': 4
            }
        }

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, realistic_temp_data):
        """Test complete feature extraction with realistic data."""
        featex = FeatExTEMP(realistic_temp_data)
        features = featex.extract_features()

        # Verify structure
        assert 'stream_1' in features
        assert len(features['stream_1']) == 5

        # Verify all features are numeric and reasonable
        for epoch_features in features['stream_1']:
            assert 30.0 < epoch_features['Mean Temperature'] < 35.0
            assert epoch_features['Min Temperature'] < epoch_features['Mean Temperature']
            assert epoch_features['Max Temperature'] > epoch_features['Mean Temperature']
            assert epoch_features['Temperature Variability'] >= 0
            assert epoch_features['Rate of Change'] >= 0

    @pytest.mark.integration
    def test_features_detect_temperature_increase(self):
        """Test that features detect increasing temperature trend."""
        # Create epochs with increasing temperature
        epochs = []
        for i in range(3):
            timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')
            base_temp = 32.0 + i * 1.0  # Each epoch 1°C warmer
            temp_data = base_temp + np.random.randn(100) * 0.1
            epoch = pd.DataFrame({'TEMP': temp_data}, index=timestamps)
            epochs.append(epoch)

        dataset = {'stream_1': {'epochs': epochs, 'sfreq': 4}}
        featex = FeatExTEMP(dataset)
        features = featex.extract_features()

        # Mean temperature should increase across epochs
        means = [f['Mean Temperature'] for f in features['stream_1']]
        assert means[1] > means[0]
        assert means[2] > means[1]

    @pytest.mark.integration
    def test_rate_of_change_detects_rapid_variation(self):
        """Test that rate of change detects rapid temperature variations."""
        # Create one stable epoch and one rapidly changing epoch
        timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')

        # Stable epoch
        stable_data = 32.0 + np.random.randn(100) * 0.05
        stable_epoch = pd.DataFrame({'TEMP': stable_data}, index=timestamps)

        # Rapidly changing epoch
        rapid_data = 32.0 + np.sin(np.linspace(0, 10, 100)) * 2
        rapid_epoch = pd.DataFrame({'TEMP': rapid_data}, index=timestamps)

        dataset = {
            'stream_1': {'epochs': [stable_epoch, rapid_epoch], 'sfreq': 4}
        }

        featex = FeatExTEMP(dataset)
        features = featex.extract_features()

        # Rapidly changing epoch should have higher rate of change
        stable_rate = features['stream_1'][0]['Rate of Change']
        rapid_rate = features['stream_1'][1]['Rate of Change']
        assert rapid_rate > stable_rate
