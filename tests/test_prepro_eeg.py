"""
Unit tests for modalities/prepro_eeg.py

Tests the PreProEEG class for EEG signal preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
import mne
from modalities.prepro_eeg import PreProEEG


class TestPreProEEG:
    """Test suite for PreProEEG class."""

    @pytest.fixture
    def sample_eeg_raw(self):
        """Create sample MNE Raw EEG data for testing."""
        n_channels = 4
        n_times = 1000
        sfreq = 256  # Hz

        # Create sample data
        data = np.random.randn(n_channels, n_times) * 50  # μV scale

        # Create channel names
        ch_names = ['AF7', 'AF8', 'TP9', 'TP10']
        ch_types = ['eeg'] * n_channels

        # Create info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create Raw object
        raw = mne.io.RawArray(data, info)

        return raw

    @pytest.fixture
    def sample_dataset(self, sample_eeg_raw):
        """Create sample dataset dictionary for PreProEEG."""
        return {
            'stream_1': {
                'data': sample_eeg_raw.copy(),
                'sfreq': 256
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test PreProEEG initialization."""
        prepro = PreProEEG(sample_dataset)
        assert prepro.dataset is not None
        assert prepro.min_sfreq == 256

    @pytest.mark.unit
    def test_apply_downsampling(self, sample_dataset):
        """Test downsampling functionality."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_downsampling(max_sfreq=128)

        assert result is not None
        assert result['stream_1']['sfreq'] == 128
        assert result['stream_1']['data'].info['sfreq'] == 128

    @pytest.mark.unit
    def test_apply_downsampling_no_change(self, sample_dataset):
        """Test downsampling when sfreq is already below max."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_downsampling(max_sfreq=512)

        # Should not change sampling rate
        assert result['stream_1']['sfreq'] == 256

    @pytest.mark.unit
    def test_apply_bandpass_filter(self, sample_dataset):
        """Test bandpass filtering."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_bandpass_filter(low_freq=1.0, high_freq=40.0)

        assert result is not None
        # Data should still exist
        assert 'data' in result['stream_1']

    @pytest.mark.unit
    def test_apply_notch_filter_single_freq(self, sample_dataset):
        """Test notch filter with single frequency."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_notch_filter(50)  # Single frequency

        assert result is not None

    @pytest.mark.unit
    def test_apply_notch_filter_multiple_freqs(self, sample_dataset):
        """Test notch filter with multiple frequencies."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_notch_filter([50, 100])  # List of frequencies

        assert result is not None

    @pytest.mark.unit
    def test_apply_epoching(self, sample_dataset):
        """Test epoching functionality."""
        prepro = PreProEEG(sample_dataset)
        result = prepro.apply_epoching(epoch_duration=1.0)  # 1-second epochs

        assert result is not None
        assert 'epochs' in result['stream_1']
        assert isinstance(result['stream_1']['epochs'], mne.Epochs)

    @pytest.mark.unit
    def test_check_normality(self, sample_dataset):
        """Test normality checking."""
        prepro = PreProEEG(sample_dataset)
        is_normal = prepro.check_normality(alpha=0.05)

        assert isinstance(is_normal, bool)

    @pytest.mark.unit
    def test_apply_normalization(self, sample_dataset):
        """Test min-max normalization."""
        prepro = PreProEEG(sample_dataset)
        prepro.apply_epoching(epoch_duration=1.0)
        result = prepro.apply_normalization()

        assert result is not None
        # Check that data was normalized
        epochs_data = result['stream_1']['epochs'].get_data()
        assert epochs_data.min() >= 0.0
        assert epochs_data.max() <= 1.0

    @pytest.mark.unit
    def test_apply_standardization(self, sample_dataset):
        """Test z-score standardization."""
        prepro = PreProEEG(sample_dataset)
        prepro.apply_epoching(epoch_duration=1.0)
        result = prepro.apply_standardization()

        assert result is not None
        # Check that data was standardized (approximately zero mean, unit variance)
        epochs_data = result['stream_1']['epochs'].get_data()
        assert np.abs(epochs_data.mean()) < 0.1  # Close to 0
        assert np.abs(epochs_data.std() - 1.0) < 0.1  # Close to 1

    @pytest.mark.unit
    def test_apply_rejection(self, sample_dataset):
        """Test epoch rejection based on amplitude threshold."""
        prepro = PreProEEG(sample_dataset)
        prepro.apply_epoching(epoch_duration=1.0)
        result = prepro.apply_rejection(threshold=100)

        assert result is not None
        assert 'epochs' in result['stream_1']

    @pytest.mark.unit
    def test_plot_eeg_data(self, sample_dataset):
        """Test EEG data plotting."""
        prepro = PreProEEG(sample_dataset)
        figs, titles = prepro.plot_eeg_data(sample_dataset, "Test Plot")

        assert len(figs) == 1
        assert len(titles) == 1
        assert "Test Plot" in titles[0]

    @pytest.mark.unit
    def test_calculate_sampling_frequency(self, sample_dataset):
        """Test sampling frequency calculation from timestamps."""
        prepro = PreProEEG(sample_dataset)

        # Create DataFrame with datetime index
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='4ms')  # 250 Hz
        data = pd.DataFrame({'value': np.random.randn(1000)}, index=timestamps)

        sfreq = prepro.calculate_sampling_frequency(data)

        # Should be close to 250 Hz
        assert 240 < sfreq < 260

    @pytest.mark.unit
    def test_calculate_sampling_frequency_invalid_index(self, sample_dataset):
        """Test sampling frequency calculation with non-datetime index."""
        prepro = PreProEEG(sample_dataset)

        # Create DataFrame without datetime index
        data = pd.DataFrame({'value': np.random.randn(100)})

        with pytest.raises(ValueError, match="must be a DatetimeIndex"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_calculate_sampling_frequency_insufficient_data(self, sample_dataset):
        """Test sampling frequency calculation with insufficient data."""
        prepro = PreProEEG(sample_dataset)

        # Create DataFrame with only 1 sample
        timestamps = pd.date_range('2024-01-01', periods=1, freq='1s')
        data = pd.DataFrame({'value': [1.0]}, index=timestamps)

        with pytest.raises(ValueError, match="not have enough data points"):
            prepro.calculate_sampling_frequency(data)

    @pytest.mark.unit
    def test_check_and_filter_valid(self, sample_eeg_raw):
        """Test _check_and_filter with valid frequency bounds."""
        prepro = PreProEEG({'test': {'data': sample_eeg_raw, 'sfreq': 256}})
        filtered = prepro._check_and_filter(sample_eeg_raw, l_freq=1.0, h_freq=40.0)

        assert filtered is not None

    @pytest.mark.unit
    def test_check_and_filter_invalid(self, sample_eeg_raw):
        """Test _check_and_filter with invalid frequency bounds."""
        prepro = PreProEEG({'test': {'data': sample_eeg_raw, 'sfreq': 256}})
        # High freq exceeds Nyquist (128 Hz)
        filtered = prepro._check_and_filter(sample_eeg_raw, l_freq=1.0, h_freq=150.0)

        # Should return original data without filtering
        assert filtered is not None


class TestPreProEEGIntegration:
    """Integration tests for PreProEEG preprocessing pipeline."""

    @pytest.fixture
    def sample_eeg_raw(self):
        """Create sample MNE Raw EEG data."""
        n_channels = 4
        n_times = 2000
        sfreq = 256

        data = np.random.randn(n_channels, n_times) * 50
        ch_names = ['AF7', 'AF8', 'TP9', 'TP10']
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        return raw

    @pytest.fixture
    def sample_dataset(self, sample_eeg_raw):
        """Create sample dataset."""
        return {
            'stream_1': {
                'data': sample_eeg_raw.copy(),
                'sfreq': 256
            }
        }

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_dataset):
        """Test complete preprocessing pipeline."""
        prepro = PreProEEG(sample_dataset)

        # Apply full pipeline
        prepro.apply_downsampling(max_sfreq=200)
        prepro.apply_notch_filter(50)
        prepro.apply_bandpass_filter(1.0, 40.0)
        prepro.apply_epoching(epoch_duration=2.0)

        is_normal = prepro.check_normality()
        if is_normal:
            prepro.apply_normalization()
        else:
            prepro.apply_standardization()

        # Verify pipeline completed
        assert 'epochs' in prepro.dataset['stream_1']
        assert prepro.dataset['stream_1']['sfreq'] == 200

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires scikit-learn for ICA (optional dependency)")
    def test_artifact_removal_pipeline(self, sample_dataset):
        """Test artifact removal with ICA (slow test, requires sklearn)."""
        prepro = PreProEEG(sample_dataset)

        # Apply preprocessing before ICA
        prepro.apply_downsampling(max_sfreq=200)
        prepro.apply_bandpass_filter(1.0, 40.0)

        # Apply artifact removal
        result = prepro.apply_artifact_removal()

        assert result is not None
        assert 'data' in result['stream_1']
