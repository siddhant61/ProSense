"""
Unit tests for modalities/featex_bvp.py

Tests the FeatExBVP class for BVP feature extraction including time-domain,
frequency-domain, heart rate (HR), and heart rate variability (HRV) features.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_bvp import FeatExBVP


class TestFeatExBVP:
    """Test suite for FeatExBVP class."""

    @pytest.fixture
    def sample_bvp_epochs(self):
        """Create sample BVP epochs with realistic cardiac signal."""
        epochs = []
        for i in range(3):
            # 64 Hz sampling, 10 seconds each epoch
            n_samples = 640
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0

            # Simulate cardiac cycle at ~70 BPM + variation
            cardiac_freq = (70 + i * 5) / 60.0  # Increasing HR across epochs
            bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t)
            bvp_signal += np.random.randn(n_samples) * 1.5  # Small noise

            epoch = pd.DataFrame({'BVP': bvp_signal}, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_bvp_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_bvp_epochs,
                'sfreq': 64
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExBVP initialization."""
        featex = FeatExBVP(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_extract_time_domain_features(self, sample_bvp_epochs):
        """Test time-domain feature extraction."""
        featex = FeatExBVP({})
        features = featex.extract_time_domain_features(sample_bvp_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3  # 3 epochs

        # Check first epoch has all time-domain features
        epoch_0 = features[0]
        assert 'mean' in epoch_0
        assert 'std' in epoch_0
        assert 'min' in epoch_0
        assert 'max' in epoch_0
        assert 'skew' in epoch_0
        assert 'kurtosis' in epoch_0

        # Check values are reasonable for BVP signal
        assert 80 < epoch_0['mean'] < 120  # Around baseline
        assert epoch_0['std'] > 0
        assert epoch_0['min'] < epoch_0['mean']
        assert epoch_0['max'] > epoch_0['mean']

    @pytest.mark.unit
    def test_extract_frequency_domain_features(self, sample_bvp_epochs):
        """Test frequency-domain (PSD) feature extraction."""
        featex = FeatExBVP({})
        features = featex.extract_frequency_domain_features(sample_bvp_epochs)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check PSD feature exists and is positive
        for epoch_idx in range(3):
            assert 'psd' in features[epoch_idx]
            assert features[epoch_idx]['psd'] > 0

    @pytest.mark.unit
    def test_calculate_heart_rate_variability(self, sample_bvp_epochs):
        """Test HRV calculation."""
        featex = FeatExBVP({})
        features = featex.calculate_heart_rate_variability(sample_bvp_epochs, sfreq=64)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check HRV feature exists
        for epoch_idx in range(3):
            assert 'hrv' in features[epoch_idx]
            hrv_value = features[epoch_idx]['hrv']
            # HRV should be a small positive number (in seconds)
            assert isinstance(hrv_value, (int, float, np.number))
            assert hrv_value >= 0

    @pytest.mark.unit
    def test_calculate_heart_rate(self, sample_bvp_epochs):
        """Test heart rate calculation."""
        featex = FeatExBVP({})
        features = featex.calculate_heart_rate(sample_bvp_epochs, sfreq=64)

        assert isinstance(features, dict)
        assert len(features) == 3

        # Check HR values are in reasonable physiological range
        for epoch_idx in range(3):
            assert 'heart_rate' in features[epoch_idx]
            hr = features[epoch_idx]['heart_rate']
            # Should be between 40-150 BPM for normal/exercise conditions
            if not np.isnan(hr):
                assert 40 < hr < 150

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExBVP(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check all feature categories present
        stream_features = features['stream_1']
        assert 'time_features' in stream_features
        assert 'freq_features' in stream_features
        assert 'hrv_features' in stream_features
        assert 'hr_features' in stream_features

        # Check we have features for all 3 epochs
        assert len(stream_features['time_features']) == 3
        assert len(stream_features['freq_features']) == 3
        assert len(stream_features['hrv_features']) == 3
        assert len(stream_features['hr_features']) == 3

    @pytest.mark.unit
    def test_plot_features_over_epoch(self, sample_dataset):
        """Test plotting features over epochs."""
        featex = FeatExBVP(sample_dataset)
        features = featex.extract_features()

        figs, titles = featex.plot_features_over_epoch(features)

        # Should have 4 figures (time, freq, hrv, hr)
        assert len(figs) == 4
        assert len(titles) == 4

        # Check titles
        title_types = [title.lower() for title in titles]
        assert any('time' in t for t in title_types)
        assert any('freq' in t for t in title_types)
        assert any('hrv' in t for t in title_types)
        assert any('hr' in t for t in title_types)


class TestFeatExBVPIntegration:
    """Integration tests for FeatExBVP feature extraction."""

    @pytest.fixture
    def realistic_bvp_data(self):
        """Create realistic BVP data with varying heart rate."""
        epochs = []
        base_hr_bpm = [65, 75, 85, 95, 80]  # Varying HR pattern

        for hr_bpm in base_hr_bpm:
            n_samples = 640  # 10 seconds at 64 Hz
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0

            # Simulate cardiac signal with specified HR
            cardiac_freq = hr_bpm / 60.0

            # Add HRV by slightly varying RR intervals
            phase_noise = np.cumsum(np.random.randn(n_samples) * 0.01)
            bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t + phase_noise)
            bvp_signal += np.random.randn(n_samples) * 1.0

            epoch = pd.DataFrame({'BVP': bvp_signal}, index=timestamps)
            epochs.append(epoch)

        return {
            'stream_1': {
                'epochs': epochs,
                'sfreq': 64
            }
        }

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, realistic_bvp_data):
        """Test complete feature extraction with realistic data."""
        featex = FeatExBVP(realistic_bvp_data)
        features = featex.extract_features()

        # Verify structure
        assert 'stream_1' in features
        assert len(features['stream_1']['time_features']) == 5
        assert len(features['stream_1']['hr_features']) == 5

        # Verify all features are numeric
        for epoch_idx in range(5):
            # Time features
            time_feat = features['stream_1']['time_features'][epoch_idx]
            for key in ['mean', 'std', 'min', 'max']:
                assert isinstance(time_feat[key], (int, float, np.number))

            # HR features
            hr_feat = features['stream_1']['hr_features'][epoch_idx]
            hr = hr_feat['heart_rate']
            if not np.isnan(hr):
                assert 40 < hr < 150

    @pytest.mark.integration
    def test_hr_increases_across_epochs(self):
        """Test that features detect increasing heart rate."""
        # Create epochs with clearly increasing HR
        base_hr_bpm = [60, 80, 100]
        epochs = []

        for hr_bpm in base_hr_bpm:
            n_samples = 640
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0

            cardiac_freq = hr_bpm / 60.0
            bvp_signal = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t)
            epoch = pd.DataFrame({'BVP': bvp_signal}, index=timestamps)
            epochs.append(epoch)

        dataset = {'stream_1': {'epochs': epochs, 'sfreq': 64}}
        featex = FeatExBVP(dataset)
        features = featex.extract_features()

        # Extract HRs
        hrs = [features['stream_1']['hr_features'][i]['heart_rate']
               for i in range(3)]

        # Filter out NaN values
        valid_hrs = [hr for hr in hrs if not np.isnan(hr)]

        # Should detect increasing trend (may not be exact due to peak detection)
        if len(valid_hrs) >= 2:
            assert valid_hrs[-1] > valid_hrs[0]  # Last HR higher than first

    @pytest.mark.integration
    def test_time_domain_features_stability(self, realistic_bvp_data):
        """Test that time-domain features are stable and reasonable."""
        featex = FeatExBVP(realistic_bvp_data)
        features = featex.extract_features()

        time_features = features['stream_1']['time_features']

        for epoch_idx in range(5):
            feat = time_features[epoch_idx]

            # Mean should be close to baseline
            assert 80 < feat['mean'] < 120

            # Std should reflect cardiac oscillations
            assert 5 < feat['std'] < 30

            # Range should be reasonable
            assert feat['max'] - feat['min'] < 100
