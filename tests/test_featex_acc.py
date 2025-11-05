"""
Unit tests for modalities/featex_acc.py

Tests the FeatExACC class for 3-axis accelerometer feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_acc import FeatExACC


class TestFeatExACC:
    """Test suite for FeatExACC class."""

    @pytest.fixture
    def sample_acc_epochs(self):
        """Create sample 3-axis accelerometer epochs."""
        epochs = []
        for i in range(3):
            n_samples = 320  # 10 seconds at 32 Hz
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')

            # Simulate movement with gravity component
            t = np.arange(n_samples) / 32.0
            x_acc = np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 0.1
            y_acc = np.cos(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 0.1
            z_acc = 9.8 + 0.5 * np.sin(2 * np.pi * 0.3 * t) + np.random.randn(n_samples) * 0.1

            epoch = pd.DataFrame({
                'X': x_acc,
                'Y': y_acc,
                'Z': z_acc
            }, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_acc_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_acc_epochs,
                'sfreq': 32
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExACC initialization."""
        featex = FeatExACC(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_extract_time_domain_features(self, sample_acc_epochs):
        """Test time-domain feature extraction."""
        featex = FeatExACC({})
        features = featex.extract_time_domain_features(sample_acc_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3  # 3 epochs

        # Check first epoch has features for all axes
        epoch_0 = features[0]
        for axis in ['x', 'y', 'z']:
            assert axis in epoch_0
            axis_features = epoch_0[axis]
            assert 'mean' in axis_features
            assert 'std' in axis_features
            assert 'min' in axis_features
            assert 'max' in axis_features
            assert 'skew' in axis_features
            assert 'kurtosis' in axis_features

            # Check std is positive
            assert axis_features['std'] > 0

    @pytest.mark.unit
    def test_extract_frequency_domain_features(self, sample_acc_epochs):
        """Test frequency-domain (PSD) feature extraction."""
        featex = FeatExACC({})
        features = featex.extract_frequency_domain_features(sample_acc_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check PSD exists for all axes
        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert axis in features[epoch_idx]
                assert 'psd' in features[epoch_idx][axis]
                assert features[epoch_idx][axis]['psd'] > 0

    @pytest.mark.unit
    def test_extract_sma(self, sample_acc_epochs):
        """Test Signal Magnitude Area extraction."""
        featex = FeatExACC({})
        features = featex.extract_sma(sample_acc_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check SMA exists and is positive
        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert 'sma' in features[epoch_idx][axis]
                assert features[epoch_idx][axis]['sma'] > 0

    @pytest.mark.unit
    def test_extract_entropy(self, sample_acc_epochs):
        """Test entropy extraction."""
        featex = FeatExACC({})
        features = featex.extract_entropy(sample_acc_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check entropy exists for all axes
        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert 'entropy' in features[epoch_idx][axis]
                # Entropy should be a real number
                assert isinstance(features[epoch_idx][axis]['entropy'], (int, float, np.number))

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExACC(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check all feature categories present
        stream_features = features['stream_1']
        assert 'time_features' in stream_features
        assert 'freq_features' in stream_features
        assert 'sma_features' in stream_features
        assert 'entropy_features' in stream_features

        # Check we have features for all 3 epochs
        assert len(stream_features['time_features']) == 3
        assert len(stream_features['freq_features']) == 3
        assert len(stream_features['sma_features']) == 3
        assert len(stream_features['entropy_features']) == 3


class TestFeatExACCIntegration:
    """Integration tests for FeatExACC feature extraction."""

    @pytest.fixture
    def activity_data(self):
        """Create ACC data simulating different activity levels."""
        epochs = []
        activity_levels = [0.5, 1.0, 2.0]  # Low, medium, high activity

        for activity in activity_levels:
            n_samples = 320
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')
            t = np.arange(n_samples) / 32.0

            # Activity intensity affects amplitude and frequency
            x_acc = activity * np.sin(2 * np.pi * activity * t) + np.random.randn(n_samples) * 0.1
            y_acc = activity * np.cos(2 * np.pi * activity * t) + np.random.randn(n_samples) * 0.1
            z_acc = 9.8 + activity * np.sin(2 * np.pi * 0.5 * activity * t) + np.random.randn(n_samples) * 0.1

            epoch = pd.DataFrame({'X': x_acc, 'Y': y_acc, 'Z': z_acc}, index=timestamps)
            epochs.append(epoch)

        return {'stream_1': {'epochs': epochs, 'sfreq': 32}}

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, activity_data):
        """Test complete feature extraction with activity data."""
        featex = FeatExACC(activity_data)
        features = featex.extract_features()

        # Verify structure
        assert 'stream_1' in features
        assert len(features['stream_1']['time_features']) == 3

        # Verify all features are numeric
        for epoch_idx in range(3):
            time_feat = features['stream_1']['time_features'][epoch_idx]
            for axis in ['x', 'y', 'z']:
                for key in ['mean', 'std', 'min', 'max']:
                    assert isinstance(time_feat[axis][key], (int, float, np.number))

    @pytest.mark.integration
    def test_activity_intensity_detection(self, activity_data):
        """Test that SMA increases with activity intensity."""
        featex = FeatExACC(activity_data)
        features = featex.extract_features()

        # Extract SMA for X-axis across epochs
        sma_x_values = [features['stream_1']['sma_features'][i]['x']['sma']
                        for i in range(3)]

        # Higher activity should have higher SMA
        assert sma_x_values[1] > sma_x_values[0]
        assert sma_x_values[2] > sma_x_values[1]

    @pytest.mark.integration
    def test_gravity_component_in_z_axis(self):
        """Test that Z-axis captures gravity component."""
        # Create stationary epoch (only gravity, no movement)
        n_samples = 320
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')

        # Stationary: X and Y near 0, Z near 9.8 m/s² (gravity)
        x_acc = np.random.randn(n_samples) * 0.05
        y_acc = np.random.randn(n_samples) * 0.05
        z_acc = 9.8 + np.random.randn(n_samples) * 0.05

        epoch = pd.DataFrame({'X': x_acc, 'Y': y_acc, 'Z': z_acc}, index=timestamps)
        dataset = {'stream_1': {'epochs': [epoch], 'sfreq': 32}}

        featex = FeatExACC(dataset)
        features = featex.extract_features()

        # Z-axis mean should be close to gravity
        z_mean = features['stream_1']['time_features'][0]['z']['mean']
        assert 9.0 < z_mean < 10.5  # Close to 9.8 m/s²

        # X and Y should be close to 0
        x_mean = features['stream_1']['time_features'][0]['x']['mean']
        y_mean = features['stream_1']['time_features'][0]['y']['mean']
        assert abs(x_mean) < 1.0
        assert abs(y_mean) < 1.0
