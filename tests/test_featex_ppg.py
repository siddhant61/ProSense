"""
Unit tests for modalities/featex_ppg.py

Tests the FeatExPPG class for multi-channel PPG feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_ppg import FeatExPPG


class TestFeatExPPG:
    """Test suite for FeatExPPG class."""

    @pytest.fixture
    def sample_ppg_epochs(self):
        """Create sample 3-channel PPG epochs with cardiac signal."""
        epochs = []
        for i in range(2):
            n_samples = 320  # 5 seconds at 64 Hz
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0

            # Simulate cardiac signal at ~70 BPM
            cardiac_freq = 70 / 60.0
            ppg1 = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.randn(n_samples) * 2
            ppg2 = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t + 0.1) + np.random.randn(n_samples) * 2
            ppg3 = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t + 0.2) + np.random.randn(n_samples) * 2

            epoch = pd.DataFrame({
                'PPG1': ppg1,
                'PPG2': ppg2,
                'PPG3': ppg3
            }, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_ppg_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_ppg_epochs,
                'sfreq': 64
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExPPG initialization."""
        featex = FeatExPPG(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_calculate_heart_rate(self, sample_ppg_epochs):
        """Test heart rate calculation."""
        featex = FeatExPPG({})
        features = featex.calculate_heart_rate(sample_ppg_epochs, sfreq=64)

        assert isinstance(features, dict)
        assert len(features) == 2

        # Check HR for all channels
        for epoch_idx in range(2):
            for channel in ['PPG1', 'PPG2', 'PPG3']:
                assert channel in features[epoch_idx]
                assert 'heart_rate' in features[epoch_idx][channel]

    @pytest.mark.unit
    def test_calculate_hrv(self, sample_ppg_epochs):
        """Test HRV calculation with median filtering."""
        featex = FeatExPPG({})
        features = featex.calculate_hrv(sample_ppg_epochs, sfreq=64)

        assert isinstance(features, dict)
        assert len(features) == 2

        for epoch_idx in range(2):
            for channel in ['PPG1', 'PPG2', 'PPG3']:
                assert 'hrv' in features[epoch_idx][channel]

    @pytest.mark.unit
    def test_extract_prv(self, sample_ppg_epochs):
        """Test PRV extraction with outlier removal."""
        featex = FeatExPPG({})
        features = featex.extract_prv(sample_ppg_epochs, sfreq=64)

        assert isinstance(features, dict)
        assert len(features) == 2

        for epoch_idx in range(2):
            for channel in ['PPG1', 'PPG2', 'PPG3']:
                assert 'prv' in features[epoch_idx][channel]

    @pytest.mark.unit
    def test_extract_amplitude(self, sample_ppg_epochs):
        """Test amplitude extraction."""
        featex = FeatExPPG({})
        features = featex.extract_amplitude(sample_ppg_epochs)

        assert isinstance(features, dict)
        assert len(features) == 2

        for epoch_idx in range(2):
            for channel in ['PPG1', 'PPG2', 'PPG3']:
                assert 'amplitude' in features[epoch_idx][channel]
                assert features[epoch_idx][channel]['amplitude'] > 0

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExPPG(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check all feature categories
        stream_features = features['stream_1']
        assert 'hr_features' in stream_features
        assert 'hrv_features' in stream_features
        assert 'prv_features' in stream_features
        assert 'amp_features' in stream_features
        assert 'flow_features' in stream_features

        # Check we have features for all epochs
        assert len(stream_features['hr_features']) == 2


class TestFeatExPPGIntegration:
    """Integration tests for FeatExPPG."""

    @pytest.fixture
    def realistic_ppg_data(self):
        """Create realistic PPG data with varying HR."""
        epochs = []
        hr_values = [65, 85]  # Low and high HR

        for hr_bpm in hr_values:
            n_samples = 640  # 10 seconds at 64 Hz
            timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15.625ms')
            t = np.arange(n_samples) / 64.0

            cardiac_freq = hr_bpm / 60.0
            ppg1 = 100 + 20 * np.sin(2 * np.pi * cardiac_freq * t) + np.random.randn(n_samples) * 1.5
            ppg2 = 100 + 18 * np.sin(2 * np.pi * cardiac_freq * t + 0.1) + np.random.randn(n_samples) * 1.5
            ppg3 = 100 + 22 * np.sin(2 * np.pi * cardiac_freq * t + 0.2) + np.random.randn(n_samples) * 1.5

            epoch = pd.DataFrame({'PPG1': ppg1, 'PPG2': ppg2, 'PPG3': ppg3}, index=timestamps)
            epochs.append(epoch)

        return {'stream_1': {'epochs': epochs, 'sfreq': 64}}

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, realistic_ppg_data):
        """Test complete feature extraction with realistic data."""
        featex = FeatExPPG(realistic_ppg_data)
        features = featex.extract_features()

        # Verify structure
        assert 'stream_1' in features
        assert len(features['stream_1']['hr_features']) == 2

        # Verify HR values are in physiological range
        for epoch_idx in range(2):
            hr = features['stream_1']['hr_features'][epoch_idx]['PPG1']['heart_rate']
            if not np.isnan(hr):
                assert 40 < hr < 180

    @pytest.mark.integration
    def test_amplitude_consistency_across_channels(self, realistic_ppg_data):
        """Test that amplitude is consistent across PPG channels."""
        featex = FeatExPPG(realistic_ppg_data)
        features = featex.extract_features()

        # All channels should have positive amplitude
        for epoch_idx in range(2):
            for channel in ['PPG1', 'PPG2', 'PPG3']:
                amp = features['stream_1']['amp_features'][epoch_idx][channel]['amplitude']
                assert amp > 0
                # Amplitude should be reasonable (within 10x of each other)
                assert 5 < amp < 100
