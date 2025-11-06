"""
Unit tests for modalities/featex_gyro.py

Tests the FeatExGYRO class for 3-axis gyroscope feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_gyro import FeatExGYRO


class TestFeatExGYRO:
    """Test suite for FeatExGYRO class."""

    @pytest.fixture
    def sample_gyro_epochs(self):
        """Create sample 3-axis gyroscope epochs."""
        epochs = []
        for i in range(3):
            n_samples = 320  # 10 seconds at 32 Hz
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')
            t = np.arange(n_samples) / 32.0

            # Simulate rotational movement
            x_gyro = 50 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 5
            y_gyro = 30 * np.cos(2 * np.pi * 0.3 * t) + np.random.randn(n_samples) * 5
            z_gyro = 20 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 5

            epoch = pd.DataFrame({'X': x_gyro, 'Y': y_gyro, 'Z': z_gyro}, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_gyro_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_gyro_epochs,
                'sfreq': 32
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExGYRO initialization."""
        featex = FeatExGYRO(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_extract_time_domain_features(self, sample_gyro_epochs):
        """Test time-domain feature extraction."""
        featex = FeatExGYRO({})
        features = featex.extract_time_domain_features(sample_gyro_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

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

    @pytest.mark.unit
    def test_extract_frequency_domain_features(self, sample_gyro_epochs):
        """Test frequency-domain (PSD) feature extraction."""
        featex = FeatExGYRO({})
        features = featex.extract_frequency_domain_features(sample_gyro_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check PSD exists for all axes
        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert axis in features[epoch_idx]
                assert 'psd' in features[epoch_idx][axis]
                assert features[epoch_idx][axis]['psd'] > 0

    @pytest.mark.unit
    def test_calculate_angular_velocity_magnitude(self, sample_gyro_epochs):
        """Test angular velocity magnitude calculation."""
        featex = FeatExGYRO({})
        features = featex.calculate_angular_velocity_magnitude(sample_gyro_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        for epoch_idx in range(3):
            assert 'angular_velocity' in features[epoch_idx]
            assert 'rate_of_change' in features[epoch_idx]
            assert 'avg_angular_velocity' in features[epoch_idx]['angular_velocity']
            # Angular velocity magnitude should be positive
            assert features[epoch_idx]['angular_velocity']['avg_angular_velocity'] >= 0

    @pytest.mark.unit
    def test_calculate_zero_crossing_rate(self, sample_gyro_epochs):
        """Test zero-crossing rate calculation."""
        featex = FeatExGYRO({})
        features = featex.calculate_zero_crossing_rate(sample_gyro_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert 'zero_crossing_rate' in features[epoch_idx][axis]
                # Zero-crossing rate should be non-negative integer
                assert features[epoch_idx][axis]['zero_crossing_rate'] >= 0

    @pytest.mark.unit
    def test_extract_spectral_energy(self, sample_gyro_epochs):
        """Test spectral energy extraction."""
        featex = FeatExGYRO({})
        features = featex.extract_spectral_energy(sample_gyro_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        for epoch_idx in range(3):
            for axis in ['x', 'y', 'z']:
                assert 'spectral_energy' in features[epoch_idx][axis]
                assert features[epoch_idx][axis]['spectral_energy'] > 0

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExGYRO(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check all feature categories
        stream_features = features['stream_1']
        assert 'time_features' in stream_features
        assert 'freq_features' in stream_features
        assert 'angular_velocity' in stream_features
        assert 'orientation_change' in stream_features
        assert 'spectral_energy' in stream_features
        assert 'zero_crossing_rate' in stream_features

        # Check we have features for all 3 epochs
        assert len(stream_features['time_features']) == 3


class TestFeatExGYROIntegration:
    """Integration tests for FeatExGYRO."""

    @pytest.fixture
    def rotation_data(self):
        """Create GYRO data simulating different rotation intensities."""
        epochs = []
        rotation_intensities = [20, 50, 100]  # Low, medium, high rotation

        for intensity in rotation_intensities:
            n_samples = 320
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')
            t = np.arange(n_samples) / 32.0

            # Rotation intensity affects amplitude
            x_gyro = intensity * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 2
            y_gyro = intensity * 0.7 * np.cos(2 * np.pi * 0.3 * t) + np.random.randn(n_samples) * 2
            z_gyro = intensity * 0.5 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 2

            epoch = pd.DataFrame({'X': x_gyro, 'Y': y_gyro, 'Z': z_gyro}, index=timestamps)
            epochs.append(epoch)

        return {'stream_1': {'epochs': epochs, 'sfreq': 32}}

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, rotation_data):
        """Test complete feature extraction with rotation data."""
        featex = FeatExGYRO(rotation_data)
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
    def test_angular_velocity_increases_with_rotation(self, rotation_data):
        """Test that angular velocity magnitude increases with rotation intensity."""
        featex = FeatExGYRO(rotation_data)
        features = featex.extract_features()

        # Extract angular velocity magnitudes
        ang_vel_values = [
            features['stream_1']['angular_velocity'][i]['angular_velocity']['avg_angular_velocity']
            for i in range(3)
        ]

        # Higher rotation should have higher angular velocity
        assert ang_vel_values[1] > ang_vel_values[0]
        assert ang_vel_values[2] > ang_vel_values[1]

    @pytest.mark.integration
    def test_stationary_gyro_near_zero(self):
        """Test that stationary gyroscope reads near zero."""
        # Create stationary epoch (no rotation, only noise)
        n_samples = 320
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='31.25ms')

        x_gyro = np.random.randn(n_samples) * 1.0  # Small noise only
        y_gyro = np.random.randn(n_samples) * 1.0
        z_gyro = np.random.randn(n_samples) * 1.0

        epoch = pd.DataFrame({'X': x_gyro, 'Y': y_gyro, 'Z': z_gyro}, index=timestamps)
        dataset = {'stream_1': {'epochs': [epoch], 'sfreq': 32}}

        featex = FeatExGYRO(dataset)
        features = featex.extract_features()

        # Mean angular velocities should be close to 0
        for axis in ['x', 'y', 'z']:
            mean_val = features['stream_1']['time_features'][0][axis]['mean']
            assert abs(mean_val) < 5.0  # Should be close to zero with only noise

    @pytest.mark.unit
    def test_infer_sensor_location(self):
        """Test sensor location inference from stream ID."""
        featex = FeatExGYRO({})

        # Test Muse sensor (should be 'head')
        location = featex.infer_sensor_location('Muses-ABCD_gyro')
        assert location == 'head'

        location = featex.infer_sensor_location('muse-1234_gyro')
        assert location == 'head'

        # Test other sensors (default to 'head' for now)
        location = featex.infer_sensor_location('empatica_e4_gyro')
        assert location == 'head'

    @pytest.mark.unit
    def test_location_specific_feature(self, rotation_data):
        """Test location-specific feature extraction."""
        featex = FeatExGYRO(rotation_data)
        epochs = rotation_data['stream_1']['epochs']

        # Test with 'head' location (should call track_orientation_angles)
        features = featex.location_specific_feature(epochs, 'head')
        assert features is not None
        assert isinstance(features, dict)
        assert 0 in features
        assert 'orientation_change' in features[0]
        assert 'rate_of_change' in features[0]

        # Test with 'unknown' location
        features = featex.location_specific_feature(epochs, 'unknown')
        assert features is None

    @pytest.mark.unit
    def test_track_orientation_angles(self, rotation_data):
        """Test orientation angle tracking."""
        featex = FeatExGYRO(rotation_data)
        epochs = rotation_data['stream_1']['epochs']

        features = featex.track_orientation_angles(epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check structure for each epoch
        for epoch_idx in range(3):
            assert 'orientation_change' in features[epoch_idx]
            assert 'rate_of_change' in features[epoch_idx]
            assert 'overall_orientation_change' in features[epoch_idx]['orientation_change']
            assert 'rate_of_change_orientation' in features[epoch_idx]['rate_of_change']

    @pytest.mark.integration
    def test_plot_features(self, rotation_data, monkeypatch):
        """Test plotting features with plot_type='ind'."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)

        featex = FeatExGYRO(rotation_data)
        features = featex.extract_features()

        # Test plot generation with 'ind' type
        plots = featex.plot_features(features, plot_type='ind')

        assert isinstance(plots, dict)
        assert 'stream_1' in plots
        assert isinstance(plots['stream_1'], list)
        assert len(plots['stream_1']) > 0

        # Close all figures
        for stream_plots in plots.values():
            for plot_item in stream_plots:
                if isinstance(plot_item, tuple):
                    figs, titles = plot_item
                    for fig in figs:
                        plt.close(fig)

    @pytest.mark.integration
    def test_plot_features_all_type(self, rotation_data, monkeypatch):
        """Test plotting features with plot_type='all'."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)

        featex = FeatExGYRO(rotation_data)
        features = featex.extract_features()

        # Test plot generation with 'all' type
        plots = featex.plot_features(features, plot_type='all')

        assert isinstance(plots, dict)
        assert 'stream_1' in plots
        assert len(plots['stream_1']) > 0

        # Close all figures
        for stream_plots in plots.values():
            for plot_item in stream_plots:
                if isinstance(plot_item, tuple):
                    figs, titles = plot_item
                    for fig in figs:
                        plt.close(fig)
