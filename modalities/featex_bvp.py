"""
Blood Volume Pulse (BVP) Feature Extraction Module

This module provides comprehensive feature extraction capabilities for BVP signals including
time-domain, frequency-domain, heart rate (HR), and heart rate variability (HRV) features.

BVP (also called PPG - Photoplethysmography) measures blood volume changes in the
microvascular bed of tissue using optical sensors. Each cardiac cycle causes blood
volume fluctuations that are captured as BVP waveform peaks.

BVP features are useful for:
- Heart rate monitoring and cardiovascular health assessment
- Heart rate variability (HRV) analysis for autonomic nervous system activity
- Stress and anxiety detection (reduced HRV indicates stress)
- Fitness and recovery monitoring
- Emotion recognition (arousal detection)
- Sleep stage classification
- Atrial fibrillation detection
- Blood pressure estimation

Signal analysis:
- Time domain: Statistical properties of BVP waveform (mean, std, skewness, kurtosis)
- Frequency domain: Spectral power distribution
- HR: Average beats per minute from peak intervals
- HRV: Variability in RR intervals (time between successive peaks)

Classes:
    FeatExBVP: Main feature extraction class for BVP signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExBVP:
    """
    Feature extraction class for Blood Volume Pulse (BVP) data.

    Extracts time-domain, frequency-domain, heart rate (HR), and heart rate
    variability (HRV) features from preprocessed BVP epochs for cardiovascular
    and autonomic nervous system analysis.

    Attributes:
        dataset (dict): Preprocessed BVP dataset with epochs

    Example:
        >>> featex = FeatExBVP(preprocessed_bvp_data)
        >>> features = featex.extract_features()
        >>> print(features['stream_1']['hrv_features'][0])  # First epoch HRV
    """
    def __init__(self, dataset):
        """
        Initialize BVP feature extractor.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {stream_id: {'epochs': [epoch_dataframes], 'sfreq': float}}
        """
        self.dataset = dataset

    def extract_time_domain_features(self, epochs):
        """
        Extract statistical time-domain features from BVP waveform.

        Computes distributional properties of the BVP signal that reflect
        waveform morphology and amplitude characteristics.

        Args:
            epochs (list): List of BVP epoch DataFrames or Series

        Returns:
            dict: Features for each epoch with structure:
                {epoch_idx: {
                    'mean': float,      # Average BVP amplitude
                    'std': float,       # Signal variability
                    'min': float,       # Minimum amplitude (diastolic)
                    'max': float,       # Maximum amplitude (systolic)
                    'skew': float,      # Asymmetry of distribution
                    'kurtosis': float   # Peakedness of distribution
                }}

        Note:
            - Mean reflects average blood volume
            - Std indicates amplitude variability
            - Skewness and kurtosis characterize waveform shape
            - Useful for detecting abnormal cardiac patterns
        """
        features = {}
        for i, epoch in enumerate(epochs):
            epoch.index = pd.to_datetime(epoch.index, unit='s')
            # If we have a DataFrame, select the column, otherwise it's just the Series
            epoch = epoch.iloc[:, 0] if isinstance(epoch, pd.DataFrame) else epoch
            features[i] = {
                'mean': np.mean(epoch),
                'std': np.std(epoch),
                'min': np.min(epoch),
                'max': np.max(epoch),
                'skew': epoch.skew(),
                'kurtosis': epoch.kurtosis()
            }
        return features

    def extract_frequency_domain_features(self, epochs):
        """
        Extract frequency-domain features using Fast Fourier Transform (FFT).

        Computes spectral power characteristics that reflect cardiac rhythm
        regularity and respiratory influences on heart rate.

        Args:
            epochs (list): List of BVP epoch DataFrames or Series

        Returns:
            dict: Features for each epoch with structure:
                {epoch_idx: {'psd': float}}  # Power spectral density

        Note:
            - PSD (Power Spectral Density) reflects overall spectral power
            - Higher PSD indicates stronger cardiac signal
            - Can be used to detect respiratory sinus arrhythmia
            - Frequency analysis complements time-domain HRV metrics
        """
        features = {}
        for i, epoch in enumerate(epochs):
            epoch.index = pd.to_datetime(epoch.index, unit='s')
            # If we have a DataFrame, select the column, otherwise it's just the Series
            epoch = epoch.iloc[:, 0] if isinstance(epoch, pd.DataFrame) else epoch
            fft_values = rfft(epoch.to_numpy())
            features[i] = {
                'psd': np.mean(np.abs(fft_values)**2)
            }
        return features

    def calculate_heart_rate_variability(self, epochs, sfreq):
        """
        Calculate Heart Rate Variability (HRV) from RR intervals.

        HRV is the variation in time between successive heartbeats (RR intervals).
        It's a key indicator of autonomic nervous system (ANS) function and adaptability.

        Args:
            epochs (list): List of BVP epoch DataFrames or Series
            sfreq (float): Sampling frequency in Hz

        Returns:
            dict: HRV for each epoch {epoch_idx: {'hrv': float}}

        Note:
            - HRV measured as standard deviation of RR intervals (SDNN)
            - Higher HRV indicates better cardiovascular health and stress resilience
            - Lower HRV associated with stress, fatigue, and poor recovery
            - Normal resting HRV: 20-200ms (varies with age and fitness)
            - Requires at least 2 detected peaks for calculation
        """
        features = {}
        for i, epoch in enumerate(epochs):
            epoch.index = pd.to_datetime(epoch.index, unit='s')
            # If we have a DataFrame, select the column, otherwise it's just the Series
            epoch = epoch.iloc[:, 0] if isinstance(epoch, pd.DataFrame) else epoch
            peaks, _ = find_peaks(epoch, distance=sfreq/2)  # Assuming at least 0.5s between heartbeats
            rr_intervals = np.diff(peaks) / sfreq
            hrv = np.std(rr_intervals)
            features[i] = {'hrv': hrv}
        return features

    def calculate_heart_rate(self, epochs, sfreq):
        """
        Calculate heart rate (HR) in beats per minute (BPM).

        Computes average heart rate from detected BVP peaks by analyzing
        time intervals between successive peaks (RR intervals).

        Args:
            epochs (list): List of BVP epoch DataFrames or Series
            sfreq (float): Sampling frequency in Hz

        Returns:
            dict: Heart rate for each epoch {epoch_idx: {'heart_rate': float}}

        Note:
            - HR = 60 / mean(RR intervals) in BPM
            - Normal resting HR: 60-100 BPM (adults)
            - Athletes may have resting HR: 40-60 BPM
            - Elevated HR indicates arousal, stress, or physical activity
            - Returns NaN if fewer than 2 peaks detected
        """
        features = {}
        for i, epoch in enumerate(epochs):
            epoch.index = pd.to_datetime(epoch.index, unit='s')
            # If we have a DataFrame, select the column, otherwise it's just the Series
            epoch = epoch.iloc[:, 0] if isinstance(epoch, pd.DataFrame) else epoch
            peaks, _ = find_peaks(epoch, distance=sfreq/2)
            rr_intervals = np.diff(peaks) / sfreq
            heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan
            features[i] = {'heart_rate': heart_rate}
        return features

    def extract_features(self):
        """
        Extract comprehensive BVP features from all epochs.

        Main entry point for BVP feature extraction. Processes all streams
        and epochs to extract time-domain, frequency-domain, HR, and HRV features.

        Returns:
            dict: Features for each stream with structure:
                {stream_id: {
                    'time_features': {epoch_idx: {'mean', 'std', 'min', 'max', 'skew', 'kurtosis'}},
                    'freq_features': {epoch_idx: {'psd'}},
                    'hrv_features': {epoch_idx: {'hrv'}},
                    'hr_features': {epoch_idx: {'heart_rate'}}
                }}

        Example:
            >>> featex = FeatExBVP(dataset)
            >>> features = featex.extract_features()
            >>> # Access HR for first epoch
            >>> hr = features['stream_1']['hr_features'][0]['heart_rate']
            >>> hrv = features['stream_1']['hrv_features'][0]['hrv']
            >>> print(f"HR: {hr:.1f} BPM, HRV: {hrv:.3f}s")

        Note:
            Each epoch yields 9 features total:
            - 6 time-domain features (statistical properties)
            - 1 frequency-domain feature (PSD)
            - 1 HRV metric (SDNN)
            - 1 HR metric (BPM)
        """
        all_features = {}
        for stream, data_info in self.dataset.items():
            epochs = data_info['epochs']
            sfreq = data_info['sfreq']

            time_features = self.extract_time_domain_features(epochs)
            freq_features = self.extract_frequency_domain_features(epochs)
            hrv_features = self.calculate_heart_rate_variability(epochs, sfreq)
            hr_features = self.calculate_heart_rate(epochs, sfreq)

            all_features[stream] = {
                'time_features': time_features,
                'freq_features': freq_features,
                'hrv_features': hrv_features,
                'hr_features': hr_features
            }

        return all_features

    def plot_features_over_epoch(self, all_features):
        """
        Plot BVP features across epochs.

        Creates separate visualizations for each feature type (time-domain,
        frequency-domain, HRV, HR) showing how cardiovascular metrics evolve.

        Args:
            all_features (dict): Extracted features from extract_features()

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Separate plot for each feature category
            - Useful for tracking cardiovascular changes over time
            - HR trends show arousal and activity levels
            - HRV trends indicate stress and recovery patterns
        """
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                # Determine the number of subplots needed
                num_features = len(epoch_features[0]) if epoch_features else 0
                fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features))
                fig.suptitle(f'{feature_type.capitalize()} - {stream}')
                title = f'{feature_type.capitalize()} - {stream}'

                if not num_features:
                    print(f"No data to plot for {feature_type} in {stream}")
                    continue

                for i, (feature_name, values) in enumerate(epoch_features[0].items()):
                    # Collect data for the feature across all epochs
                    feature_values = [features.get(feature_name, np.nan) for features in epoch_features.values()]

                    # Plot the data
                    ax = axes[i] if num_features > 1 else axes
                    ax.plot(feature_values, label=feature_name)
                    ax.set_title(f'{feature_name}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(f'{feature_type.capitalize()} Value')
                    ax.legend(loc='upper left')

                plt.tight_layout()
                figs.append(fig)
                titles.append(title)
        return figs, titles

    def plot_features_over_time(self, all_features, epoch_duration=5):
        """
        Plot BVP features over absolute time.

        Creates time-series visualizations with real timestamps for correlating
        cardiovascular metrics with external events and activities.

        Args:
            all_features (dict): Extracted features from extract_features()
            epoch_duration (float): Length of each epoch in seconds (default: 5)

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - X-axis shows absolute timestamps
            - Useful for correlating HR/HRV with activities or stressors
            - Can reveal circadian patterns in resting heart rate
            - HRV dips visible during stress periods
            - HR spikes indicate arousal or physical activity
        """
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            start_time = pd.to_datetime(self.dataset[stream]['epochs'][0].index[0], unit='s')

            for feature_type, epoch_features in feature_data.items():
                num_features = len(epoch_features[0]) if epoch_features else 0
                fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features))
                fig.suptitle(f'{feature_type.capitalize()} - {stream}')
                title = f'{feature_type.capitalize()} - {stream}'

                if not num_features:
                    print(f"No data to plot for {feature_type} in {stream}")
                    continue

                for i, (feature_name, values) in enumerate(epoch_features[0].items()):
                    # Collect data for each feature across all epochs
                    times_vals = [(start_time + pd.to_timedelta(epoch_idx * epoch_duration, unit='s'),
                                   features.get(feature_name, np.nan)) for epoch_idx, features in
                                  enumerate(epoch_features.values())]
                    times, vals = zip(*times_vals)

                    # Plot the data
                    ax = axes[i] if num_features > 1 else axes
                    ax.plot(times, vals, label=feature_name)
                    ax.set_title(f'{feature_name}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'{feature_type.capitalize()} Value')
                    ax.legend(loc='upper left')

                    # Tilt the x-axis labels
                    for label in ax.get_xticklabels():
                        label.set_rotation(15)
                        label.set_ha('right')

                plt.tight_layout()
                figs.append(fig)
                titles.append(title)
        return figs, titles





