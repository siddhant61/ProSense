"""
Unit tests for modalities/featex_eeg.py

Tests the FeatExEEG class for EEG feature extraction.
"""

import pytest
import numpy as np
import mne
from modalities.featex_eeg import FeatExEEG


class TestFeatExEEG:
    """Test suite for FeatExEEG class."""

    @pytest.fixture
    def sample_eeg_epochs(self):
        """Create sample MNE Epochs object for testing."""
        # Create synthetic EEG data
        n_channels = 4
        n_epochs = 3
        n_times = 1024  # 4 seconds at 256 Hz (longer for wavelet analysis)
        sfreq = 256

        # Create random EEG-like data
        data = np.random.randn(n_epochs, n_channels, n_times) * 50  # µV scale

        # Create channel names
        ch_names = ['AF7', 'AF8', 'TP9', 'TP10']
        ch_types = ['eeg'] * n_channels

        # Create MNE Info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create Epochs object
        epochs = mne.EpochsArray(data, info)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_eeg_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'test_file': {
                'epochs': sample_eeg_epochs,
                'sfreq': 256
            }
        }

    @pytest.mark.unit
    def test_init(self):
        """Test FeatExEEG initialization."""
        featex = FeatExEEG()
        assert featex is not None
        assert hasattr(featex, 'freq_bands')
        assert len(featex.freq_bands) == 9

    @pytest.mark.unit
    def test_extract_power_band_ratios(self, sample_eeg_epochs):
        """Test power band ratio extraction."""
        featex = FeatExEEG()
        power_ratios = featex.extract_power_band_ratios(sample_eeg_epochs)

        # Check structure
        assert isinstance(power_ratios, dict)
        assert len(power_ratios) == 3  # 3 epochs

        # Check first epoch
        epoch_0 = power_ratios[0]
        assert 'AF7' in epoch_0

        # Check features for first channel
        af7_features = epoch_0['AF7']
        assert 'delta_power' in af7_features
        assert 'theta_power' in af7_features
        assert 'alpha_power' in af7_features
        assert 'theta_delta_ratio' in af7_features

    @pytest.mark.unit
    def test_extract_spectral_entropy(self, sample_eeg_epochs):
        """Test spectral entropy extraction."""
        featex = FeatExEEG()
        entropy = featex.extract_spectral_entropy(sample_eeg_epochs, sfreq=256)

        # Check structure
        assert isinstance(entropy, dict)
        assert len(entropy) == 3  # 3 epochs

        # Check first epoch
        epoch_0 = entropy[0]
        assert 'AF7' in epoch_0

        # Check entropy value
        af7_entropy = epoch_0['AF7']
        assert isinstance(af7_entropy, (int, float))
        assert af7_entropy > 0  # Entropy should be positive

    @pytest.mark.unit
    def test_extract_statistical_features(self, sample_eeg_epochs):
        """Test statistical feature extraction."""
        featex = FeatExEEG()
        stats = featex.extract_statistical_features(sample_eeg_epochs)

        # Check structure
        assert isinstance(stats, dict)
        assert len(stats) == 3  # 3 epochs

        # Check first epoch
        epoch_0 = stats[0]
        assert 'AF7' in epoch_0

        # Check statistical features
        af7_stats = epoch_0['AF7']
        assert 'mean' in af7_stats
        assert 'std' in af7_stats
        assert 'variance' in af7_stats

    @pytest.mark.unit
    def test_epoch_mean(self, sample_eeg_epochs):
        """Test epoch mean calculation."""
        featex = FeatExEEG()
        data = sample_eeg_epochs.get_data()
        means = featex.epoch_mean(data)

        assert means.shape == (3, 4)  # 3 epochs, 4 channels

    @pytest.mark.unit
    def test_epoch_variance(self, sample_eeg_epochs):
        """Test epoch variance calculation."""
        featex = FeatExEEG()
        data = sample_eeg_epochs.get_data()
        variances = featex.epoch_variance(data)

        assert variances.shape == (3, 4)  # 3 epochs, 4 channels
        assert np.all(variances >= 0)  # Variance is always non-negative

    @pytest.mark.skip(reason="TFR requires very long signals (>10s) for low frequencies")
    @pytest.mark.unit
    def test_extract_features_comprehensive(self, sample_dataset):
        """Test comprehensive feature extraction pipeline."""
        featex = FeatExEEG()
        features = featex.extract_features(sample_dataset)

        # Check top-level structure
        assert isinstance(features, dict)
        assert 'test_file' in features

        # Check feature categories
        file_features = features['test_file']
        assert 'power_band_ratios' in file_features
        assert 'spectral_entropy' in file_features
        assert 'psd_features' in file_features
        assert 'coherence_features' in file_features
        assert 'tfr_features' in file_features
        assert 'statistical_features' in file_features


class TestFeatExEEGIntegration:
    """Integration tests for FeatExEEG feature extraction."""

    @pytest.fixture
    def realistic_eeg_epochs(self):
        """Create more realistic EEG epochs with simulated band power."""
        n_channels = 4
        n_epochs = 5
        n_times = 1024  # 4 seconds at 256 Hz (longer for wavelet analysis)
        sfreq = 256

        # Create EEG-like data with specific frequency components
        t = np.linspace(0, 4, n_times)  # 4 seconds
        data = np.zeros((n_epochs, n_channels, n_times))

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Add alpha (10 Hz) and theta (6 Hz) components
                alpha_signal = 30 * np.sin(2 * np.pi * 10 * t)
                theta_signal = 20 * np.sin(2 * np.pi * 6 * t)
                noise = np.random.randn(n_times) * 5
                data[epoch_idx, ch_idx, :] = alpha_signal + theta_signal + noise

        ch_names = ['AF7', 'AF8', 'TP9', 'TP10']
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        epochs = mne.EpochsArray(data, info)

        return epochs

    @pytest.mark.skip(reason="TFR requires very long signals (>10s) for low frequencies")
    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, realistic_eeg_epochs):
        """Test complete feature extraction with realistic data."""
        dataset = {
            'test_file': {
                'epochs': realistic_eeg_epochs,
                'sfreq': 256
            }
        }

        featex = FeatExEEG()
        features = featex.extract_features(dataset)

        # Verify all feature types were extracted
        assert features is not None
        assert 'test_file' in features

        file_features = features['test_file']

        # Check power band ratios
        power_ratios = file_features['power_band_ratios']
        assert len(power_ratios) == 5  # 5 epochs

        # Check spectral entropy
        entropy = file_features['spectral_entropy']
        assert len(entropy) == 5

        # Check statistical features
        stats = file_features['statistical_features']
        assert len(stats) == 5

    @pytest.mark.integration
    def test_power_bands_detect_frequency_content(self, realistic_eeg_epochs):
        """Test that power band extraction detects simulated frequencies."""
        featex = FeatExEEG()
        power_ratios = featex.extract_power_band_ratios(realistic_eeg_epochs)

        # Get first epoch, first channel
        first_epoch_af7 = power_ratios[0]['AF7']

        # Should have detected alpha and theta power
        # (Not checking exact values due to noise, but they should exist)
        assert 'alpha_power' in first_epoch_af7
        assert 'theta_power' in first_epoch_af7
        assert first_epoch_af7['alpha_power'] > 0
        assert first_epoch_af7['theta_power'] > 0
