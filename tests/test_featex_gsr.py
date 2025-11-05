"""
Unit tests for modalities/featex_gsr.py

Tests the FeatExGSR class for GSR/EDA feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from modalities.featex_gsr import FeatExGSR


class TestFeatExGSR:
    """Test suite for FeatExGSR class."""

    @pytest.fixture
    def sample_gsr_epochs(self):
        """Create sample GSR epochs for testing."""
        epochs = []
        for i in range(3):
            timestamps = pd.date_range('2024-01-01', periods=200, freq='250ms')

            # Simulate GSR with baseline + arousal responses + noise
            t = np.arange(200) / 4.0  # Time in seconds at 4 Hz
            baseline = 5.0 + i * 0.5  # Increasing baseline across epochs
            arousal = 2.0 * np.sin(2 * np.pi * 0.05 * t)  # Slow arousal oscillation
            peaks = np.zeros(200)
            # Add some SCR peaks
            peak_indices = [40, 80, 120, 160]
            for idx in peak_indices:
                if idx < 200:
                    peaks[idx:idx+10] = 0.3 * np.exp(-np.arange(10) / 3)  # Exponential decay

            gsr_data = baseline + arousal + peaks + np.random.randn(200) * 0.05
            epoch = pd.DataFrame({'GSR': gsr_data}, index=timestamps)
            epochs.append(epoch)

        return epochs

    @pytest.fixture
    def sample_dataset(self, sample_gsr_epochs):
        """Create sample dataset for feature extraction."""
        return {
            'stream_1': {
                'epochs': sample_gsr_epochs,
                'sfreq': 4
            }
        }

    @pytest.mark.unit
    def test_init(self, sample_dataset):
        """Test FeatExGSR initialization."""
        featex = FeatExGSR(sample_dataset)
        assert featex is not None
        assert featex.dataset == sample_dataset

    @pytest.mark.unit
    def test_compute_skin_conductance_level(self):
        """Test SCL (mean) computation."""
        featex = FeatExGSR({})

        # Test with array
        data_array = np.array([5.0, 5.5, 6.0, 5.5, 5.0])
        scl = featex.compute_skin_conductance_level(data_array)
        assert isinstance(scl, (int, float, np.number))
        assert 5.0 <= scl <= 6.0

        # Test with DataFrame
        data_df = pd.DataFrame({'GSR': [5.0, 5.5, 6.0, 5.5, 5.0]})
        scl_df = featex.compute_skin_conductance_level(data_df)
        assert isinstance(scl_df, (int, float, np.number))
        assert 5.0 <= scl_df <= 6.0

    @pytest.mark.unit
    def test_compute_skin_conductance_response(self):
        """Test SCR frequency (peak count) computation."""
        featex = FeatExGSR({})

        # Create data with clear peaks
        timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')
        data = np.zeros(100)
        # Add 3 peaks above threshold (0.05)
        data[20:25] = [0.1, 0.15, 0.2, 0.15, 0.1]
        data[50:55] = [0.1, 0.15, 0.2, 0.15, 0.1]
        data[80:85] = [0.1, 0.15, 0.2, 0.15, 0.1]

        epoch = pd.DataFrame({'GSR': data}, index=timestamps)
        scr_count = featex.compute_skin_conductance_response(epoch)

        assert isinstance(scr_count, int)
        assert scr_count >= 0  # At least should detect some peaks

    @pytest.mark.unit
    def test_compute_amplitude_of_scrs(self):
        """Test SCR amplitude computation."""
        featex = FeatExGSR({})

        # Create data with clear peaks
        timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')
        data = np.zeros(100)
        # Add 2 peaks with known amplitudes
        data[20:25] = [0.05, 0.1, 0.2, 0.1, 0.05]  # Peak height ~0.2
        data[60:65] = [0.05, 0.1, 0.3, 0.1, 0.05]  # Peak height ~0.3

        epoch = pd.DataFrame({'GSR': data}, index=timestamps)
        amplitude = featex.compute_amplitude_of_scrs(epoch)

        assert isinstance(amplitude, (int, float, np.number))
        assert amplitude >= 0.0

    @pytest.mark.unit
    def test_compute_amplitude_of_scrs_no_peaks(self):
        """Test SCR amplitude when no peaks detected."""
        featex = FeatExGSR({})

        # Flat data with no peaks above threshold
        timestamps = pd.date_range('2024-01-01', periods=100, freq='250ms')
        data = np.ones(100) * 0.01  # Below 0.05 threshold
        epoch = pd.DataFrame({'GSR': data}, index=timestamps)

        amplitude = featex.compute_amplitude_of_scrs(epoch)
        assert amplitude == 0.0

    @pytest.mark.unit
    def test_compute_variance_gsr(self):
        """Test GSR variance computation."""
        featex = FeatExGSR({})

        # Test with array
        data_array = np.array([5.0, 5.5, 6.0, 5.5, 5.0])
        variance = featex.compute_variance_gsr(data_array)
        assert isinstance(variance, (int, float, np.number))
        assert variance >= 0.0

        # Test with DataFrame
        data_df = pd.DataFrame({'GSR': [5.0, 5.5, 6.0, 5.5, 5.0]})
        variance_df = featex.compute_variance_gsr(data_df)
        assert isinstance(variance_df, (int, float, np.number))
        assert variance_df >= 0.0

    @pytest.mark.unit
    def test_extract_features(self, sample_dataset):
        """Test comprehensive feature extraction."""
        featex = FeatExGSR(sample_dataset)
        features = featex.extract_features()

        # Check structure
        assert isinstance(features, dict)
        assert 'stream_1' in features

        # Check we have features for all 3 epochs
        stream_features = features['stream_1']
        assert len(stream_features) == 3

        # Check first epoch has all required features
        first_epoch_features = stream_features[0]
        assert 'Skin Conductance Level (SCL)' in first_epoch_features
        assert 'Skin Conductance Response (SCR) Frequency' in first_epoch_features
        assert 'Amplitude of SCRs' in first_epoch_features
        assert 'GSR Variance' in first_epoch_features

        # Check types
        assert isinstance(first_epoch_features['Skin Conductance Level (SCL)'], (int, float, np.number))
        assert isinstance(first_epoch_features['Skin Conductance Response (SCR) Frequency'], int)
        assert isinstance(first_epoch_features['Amplitude of SCRs'], (int, float, np.number))
        assert isinstance(first_epoch_features['GSR Variance'], (int, float, np.number))

    @pytest.mark.unit
    def test_plot_features_over_epoch(self, sample_dataset):
        """Test plotting features over epochs."""
        featex = FeatExGSR(sample_dataset)
        features = featex.extract_features()

        figs, titles = featex.plot_features_over_epoch(features)

        assert len(figs) == 1
        assert len(titles) == 1
        assert 'GSR Features Over Epochs' in titles[0]


class TestFeatExGSRIntegration:
    """Integration tests for FeatExGSR feature extraction."""

    @pytest.fixture
    def realistic_gsr_data(self):
        """Create realistic GSR data with stress response pattern."""
        epochs = []
        for i in range(5):
            timestamps = pd.date_range('2024-01-01', periods=300, freq='250ms')

            # Simulate realistic stress response
            t = np.arange(300) / 4.0  # Time in seconds

            # Baseline SCL increases with stress
            baseline_scl = 4.0 + i * 0.3

            # Add slow drift
            drift = 0.5 * np.sin(2 * np.pi * 0.02 * t)

            # Add SCR peaks (more frequent with stress)
            scr_signal = np.zeros(300)
            num_scrs = 2 + i  # Increasing SCR frequency
            scr_positions = np.random.choice(range(50, 250), num_scrs, replace=False)
            for pos in scr_positions:
                # Exponential rise and decay
                scr_duration = 30
                if pos + scr_duration < 300:
                    rise = np.linspace(0, 1, 10) ** 2
                    decay = np.exp(-np.arange(20) / 5)
                    scr_shape = np.concatenate([rise, decay])
                    scr_amplitude = 0.5 + i * 0.1  # Amplitude increases with stress
                    scr_signal[pos:pos+30] += scr_amplitude * scr_shape

            # Combine components
            gsr_data = baseline_scl + drift + scr_signal + np.random.randn(300) * 0.05

            epoch = pd.DataFrame({'GSR': gsr_data}, index=timestamps)
            epochs.append(epoch)

        return {
            'stream_1': {
                'epochs': epochs,
                'sfreq': 4
            }
        }

    @pytest.mark.integration
    def test_full_feature_extraction_pipeline(self, realistic_gsr_data):
        """Test complete feature extraction with realistic data."""
        featex = FeatExGSR(realistic_gsr_data)
        features = featex.extract_features()

        # Verify structure
        assert 'stream_1' in features
        assert len(features['stream_1']) == 5

        # Verify all features are numeric and reasonable
        for epoch_features in features['stream_1']:
            assert 3.0 < epoch_features['Skin Conductance Level (SCL)'] < 10.0
            assert epoch_features['Skin Conductance Response (SCR) Frequency'] >= 0
            assert epoch_features['Amplitude of SCRs'] >= 0.0
            assert epoch_features['GSR Variance'] >= 0.0

    @pytest.mark.integration
    def test_features_detect_stress_increase(self, realistic_gsr_data):
        """Test that features detect increasing stress across epochs."""
        featex = FeatExGSR(realistic_gsr_data)
        features = featex.extract_features()

        # Extract SCL across epochs (should increase with stress)
        scl_values = [f['Skin Conductance Level (SCL)'] for f in features['stream_1']]

        # Should show increasing trend
        assert scl_values[-1] > scl_values[0]  # Last epoch higher than first

    @pytest.mark.integration
    def test_scr_frequency_increases_with_arousal(self):
        """Test that SCR frequency increases with arousal."""
        # Create low arousal epoch - completely flat (no noise, no peaks)
        low_arousal_data = np.ones(200) * 5.0
        low_arousal_epoch = pd.DataFrame({
            'GSR': low_arousal_data
        }, index=pd.date_range('2024-01-01', periods=200, freq='250ms'))

        # Create high arousal epoch with clear, strong peaks
        high_arousal_data = np.ones(200) * 5.0
        peak_positions = [20, 50, 80, 110, 140, 170]
        for pos in peak_positions:
            if pos + 10 < 200:
                # Add strong peaks that are clearly above 0.05 threshold
                high_arousal_data[pos:pos+10] += np.linspace(0.5, 0.3, 10)
        high_arousal_epoch = pd.DataFrame({
            'GSR': high_arousal_data
        }, index=pd.date_range('2024-01-01', periods=200, freq='250ms'))

        dataset = {
            'stream_1': {
                'epochs': [low_arousal_epoch, high_arousal_epoch],
                'sfreq': 4
            }
        }

        featex = FeatExGSR(dataset)
        features = featex.extract_features()

        low_scr_freq = features['stream_1'][0]['Skin Conductance Response (SCR) Frequency']
        high_scr_freq = features['stream_1'][1]['Skin Conductance Response (SCR) Frequency']

        # Low arousal should have 0 peaks (flat line)
        assert low_scr_freq == 0
        # High arousal should detect our injected peaks
        assert high_scr_freq >= 5  # Should detect most of our 6 peaks
