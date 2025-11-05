"""
Accelerometer (ACC) Feature Extraction Module

This module provides comprehensive feature extraction capabilities for 3-axis accelerometer
signals including time-domain, frequency-domain, signal magnitude area (SMA), and entropy features.

Accelerometer signals measure acceleration forces in three axes (X, Y, Z), typically in
units of m/s² or g (1g ≈ 9.81 m/s²). These signals capture body movements, orientation,
and activity patterns.

Accelerometer features are useful for:
- Activity recognition (walking, running, sitting, standing, climbing stairs)
- Fall detection and prevention
- Gait analysis and balance assessment
- Energy expenditure estimation
- Sleep quality monitoring (movement during sleep)
- Gesture recognition and human-computer interaction
- Posture classification
- Sports performance analysis
- Parkinson's disease tremor detection

Signal components:
- Static acceleration: Gravity component indicating device orientation
- Dynamic acceleration: Body movement component
- SMA (Signal Magnitude Area): Overall activity intensity metric
- Entropy: Movement complexity and unpredictability

Classes:
    FeatExACC: Main feature extraction class for 3-axis accelerometer signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExACC:
    """
    Feature extraction class for 3-axis Accelerometer (ACC) data.

    Extracts time-domain, frequency-domain, SMA, and entropy features from
    preprocessed accelerometer epochs for activity recognition and movement analysis.

    Attributes:
        dataset (dict): Preprocessed accelerometer dataset with epochs

    Example:
        >>> featex = FeatExACC(preprocessed_acc_data)
        >>> features = featex.extract_features()
        >>> print(features['stream_1']['time_features'][0]['x'])  # X-axis features
    """
    def __init__(self, dataset):
        """
        Initialize ACC feature extractor.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {stream_id: {'epochs': [epoch_dataframes], 'sfreq': float}}
        """
        self.dataset = dataset

    def extract_time_domain_features(self, epochs):
        """
        Extract statistical time-domain features from each accelerometer axis.

        Args:
            epochs (list): List of accelerometer epoch DataFrames (3 columns: X, Y, Z)

        Returns:
            dict: Features for each epoch and axis:
                {epoch_idx: {'x': {features}, 'y': {features}, 'z': {features}}}

        Features per axis:
            - mean: Average acceleration (reflects orientation/gravity)
            - std: Acceleration variability (movement intensity)
            - min/max: Range of motion
            - skew: Asymmetry of movement
            - kurtosis: Peakedness (abrupt vs smooth movements)
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            # Extract features for each channel
            for j, channel in enumerate(['x', 'y', 'z']):
                channel_features = {
                    'mean': np.mean(epoch.iloc[:, j]),
                    'std': np.std(epoch.iloc[:, j]),
                    'min': np.min(epoch.iloc[:, j]),
                    'max': np.max(epoch.iloc[:, j]),
                    'skew': skew(epoch.iloc[:, j]),
                    'kurtosis': kurtosis(epoch.iloc[:, j])
                }
                features[i][channel] = channel_features
        return features

    def extract_frequency_domain_features(self, epochs):
        """
        Extract frequency-domain features using FFT for each axis.

        Args:
            epochs (list): List of accelerometer epoch DataFrames

        Returns:
            dict: PSD for each epoch and axis

        Note:
            - PSD reflects movement frequency characteristics
            - Higher PSD in low frequencies: slow movements (walking)
            - Higher PSD in high frequencies: rapid movements (running)
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            # Extract features for each channel
            for j, channel in enumerate(['x', 'y', 'z']):
                fft_values = rfft(epoch.iloc[:, j].to_numpy())
                channel_features = {
                    'psd': np.mean(np.abs(fft_values)**2)
                }
                features[i][channel] = channel_features
        return features

    def extract_sma(self, epochs):
        """
        Extract Signal Magnitude Area (SMA) - overall activity intensity metric.

        SMA = sum of absolute acceleration across all axes, indicating total
        movement/activity level regardless of direction.

        Args:
            epochs (list): List of accelerometer epoch DataFrames

        Returns:
            dict: SMA for each epoch and axis

        Note:
            - Higher SMA = more intense activity
            - Useful for distinguishing sedentary vs active states
            - Common in activity recognition algorithms
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            # Extract features for each channel
            for j, channel in enumerate(['x', 'y', 'z']):
                channel_features = {
                    'sma': np.sum(np.sum(np.abs(epoch), axis=1))
                }
                features[i][channel] = channel_features
        return features

    def extract_entropy(self, epochs):
        """
        Extract entropy - measure of signal complexity/unpredictability.

        Args:
            epochs (list): List of accelerometer epoch DataFrames

        Returns:
            dict: Entropy for each epoch and axis

        Note:
            - Higher entropy = more complex/irregular movements
            - Lower entropy = more repetitive/periodic movements
            - Useful for detecting irregular gait or tremors
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            # Extract features for each channel
            for j, channel in enumerate(['x', 'y', 'z']):
                channel_features = {
                    'entropy': -np.sum(np.log2(epoch.iloc[:, j] + 1e-10) * (epoch.iloc[:, j] + 1e-10))
                }
                features[i][channel] = channel_features
        return features

    def extract_features(self):
        """
        Extract comprehensive accelerometer features from all epochs.

        Main entry point for ACC feature extraction. Processes all streams
        and epochs to extract time-domain, frequency-domain, SMA, and entropy features.

        Returns:
            dict: Features for each stream with structure:
                {stream_id: {
                    'time_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'freq_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'sma_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'entropy_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}}
                }}

        Note:
            Each epoch yields 30 features total:
            - 18 time-domain features (6 per axis × 3 axes)
            - 3 frequency-domain features (PSD per axis)
            - 3 SMA features (per axis)
            - 3 entropy features (per axis)
        """
        all_features = {}

        for stream, data in self.dataset.items():
            sfreq = data['sfreq']
            epochs = data['epochs']

            time_features = self.extract_time_domain_features(epochs)
            freq_features = self.extract_frequency_domain_features(epochs)
            sma_features = self.extract_sma(epochs)
            entropy_features = self.extract_entropy(epochs)

            # Combine all features into a single dictionary for the file
            all_features[stream] = {
                "time_features": time_features,
                "freq_features": freq_features,
                "sma_features": sma_features,
                "entropy_features": entropy_features
            }

        return all_features

    def plot_features_over_epoch(self, all_features):
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                num_epochs = len(epoch_features)

                # Iterate over each feature within the channel
                for feature in epoch_features[0]['x'].keys():
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                    fig.suptitle(f'{stream} - {feature.capitalize()} over Epochs', fontsize=16)
                    title = f'{stream} - {feature.capitalize()} over Epochs'
                    for j, channel in enumerate(['x', 'y', 'z']):
                        channel_feature_values = [epoch_features[i][channel][feature] for i in range(num_epochs)]

                        ax = axes[j]
                        ax.plot(range(num_epochs), channel_feature_values, label=f'{channel.upper()} Channel - {feature.capitalize()}')
                        ax.set_title(f'{channel.upper()} Channel')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel(f'{feature.capitalize()} Value')
                        ax.legend()

                    plt.tight_layout()
                    figs.append(fig)
                    titles.append(title)
        return figs, titles

    def plot_features_over_time(self, all_features, epoch_duration=5):
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                # Iterate over each feature within the channel
                for feature in epoch_features[0]['x'].keys():
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                    fig.suptitle(f'{stream} - {feature.capitalize()} over Time', fontsize=16)
                    title = f'{stream} - {feature.capitalize()} over Time'
                    for j, channel in enumerate(['x', 'y', 'z']):
                        # Prepare lists to hold the median timestamps and feature values
                        median_time_points = []
                        channel_feature_values = []

                        # Collect median timestamps and feature values
                        for epoch_idx, epoch in enumerate(self.dataset[stream]['epochs']):
                            # Convert epoch index to timestamps
                            timestamps = pd.to_datetime(epoch.index, unit='s')
                            median_timestamp = timestamps[0]
                            median_time_points.append(median_timestamp)
                            # Append feature value for this epoch
                            channel_feature_values.append(epoch_features[epoch_idx][channel][feature])

                        ax = axes[j]
                        ax.plot(median_time_points, channel_feature_values,
                                label=f'{channel.upper()} Channel - {feature.capitalize()}')
                        ax.set_title(f'{channel.upper()} Channel')
                        ax.set_xlabel('Time')
                        ax.set_ylabel(f'{feature.capitalize()} Value')

                        # # Format the x-axis to display time properly
                        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                        # ax.xaxis.set_major_locator(mdates.AutoDateLocator())

                        # Optionally rotate the x-axis labels for better readability
                        for label in ax.get_xticklabels():
                            label.set_rotation(45)
                            label.set_ha('right')

                        ax.legend()
                    plt.tight_layout()
                    figs.append(fig)
                    titles.append(title)
        return figs, titles



