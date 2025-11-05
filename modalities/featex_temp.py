"""
Temperature (TEMP) Feature Extraction Module

This module provides feature extraction capabilities for skin temperature signals
extracting statistical and variability features for thermal analysis.

Temperature features are useful for:
- Stress detection (temperature drops during stress)
- Sleep quality assessment (temperature variations during sleep cycles)
- Circadian rhythm analysis
- Emotional state detection
- Physical activity monitoring
- Fever and illness detection
- Thermoregulation assessment

Features extracted:
- Mean temperature (baseline thermal state)
- Min/Max temperature (range of thermal variation)
- Temperature variability (standard deviation)
- Rate of change (thermal dynamics)

Classes:
    FeatExTEMP: Main feature extraction class for temperature signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class FeatExTEMP:
    """
    Feature extraction class for skin temperature data.

    Extracts statistical features from preprocessed temperature epochs
    for thermal state analysis and monitoring.

    Attributes:
        dataset (dict): Preprocessed temperature dataset with epochs

    Example:
        >>> featex = FeatExTEMP(preprocessed_temp_data)
        >>> features = featex.extract_features()
        >>> print(features['stream_1'][0])  # First epoch features
    """

    def __init__(self, dataset):
        """
        Initialize temperature feature extractor.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {stream_id: {'epochs': [epoch_dataframes], 'sfreq': float}}
        """
        self.dataset = dataset

    def compute_mean_temperature(self, data):
        """
        Compute mean temperature for an epoch.

        Args:
            data (array-like): Temperature data for one epoch

        Returns:
            float: Mean temperature value

        Note:
            Represents baseline thermal state during the epoch
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.mean(data))

    def compute_max_temperature(self, data):
        """
        Compute maximum temperature for an epoch.

        Args:
            data (array-like): Temperature data for one epoch

        Returns:
            float: Maximum temperature value

        Note:
            Useful for detecting thermal peaks (e.g., physical activity)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.max(data))

    def compute_min_temperature(self, data):
        """
        Compute minimum temperature for an epoch.

        Args:
            data (array-like): Temperature data for one epoch

        Returns:
            float: Minimum temperature value

        Note:
            Useful for detecting temperature drops (e.g., stress response)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.min(data))

    def compute_temperature_variability(self, data):
        """
        Compute temperature variability (standard deviation).

        Args:
            data (array-like): Temperature data for one epoch

        Returns:
            float: Standard deviation of temperature

        Note:
            Higher variability may indicate unstable thermal state or arousal
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.std(data))

    def compute_rate_of_change(self, data):
        """
        Compute mean absolute rate of temperature change.

        Args:
            data (array-like): Temperature data for one epoch

        Returns:
            float: Mean absolute derivative (°C per sample)

        Note:
            - Measures thermal dynamics (how fast temperature changes)
            - Higher values indicate rapid thermal transitions
            - Returns 0 for single-sample epochs
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        if len(data) > 1:
            derivative = np.diff(data)
            return float(np.mean(np.abs(derivative)))
        else:
            return 0.0

    def extract_features(self):
        """
        Extract comprehensive temperature features from all epochs.

        Main entry point for temperature feature extraction. Processes all
        streams and epochs to extract statistical thermal features.

        Returns:
            dict: Features for each stream with structure:
                {stream_id: [
                    {
                        'Min Temperature': float,
                        'Mean Temperature': float,
                        'Max Temperature': float,
                        'Temperature Variability': float,
                        'Rate of Change': float
                    },
                    ...  # One dict per epoch
                ]}

        Example:
            >>> featex = FeatExTEMP(dataset)
            >>> features = featex.extract_features()
            >>> # Access features for first epoch of stream_1
            >>> epoch_0_features = features['stream_1'][0]
            >>> print(f"Mean temp: {epoch_0_features['Mean Temperature']}")

        Note:
            Each epoch yields 5 features for machine learning pipelines
        """
        all_features = {}
        for stream, data_info in self.dataset.items():
            epochs = data_info['epochs']
            stream_features = []

            for epoch in epochs:
                features = {
                    "Min Temperature": self.compute_min_temperature(epoch),
                    "Mean Temperature": self.compute_mean_temperature(epoch),
                    "Max Temperature": self.compute_max_temperature(epoch),
                    "Temperature Variability": self.compute_temperature_variability(epoch),
                    "Rate of Change": self.compute_rate_of_change(epoch)
                }
                stream_features.append(features)

            all_features[stream] = stream_features

        return all_features

    def plot_features_over_epoch(self, all_features):
        figs = []
        titles = []
        for stream, epoch_features in all_features.items():
            num_features = len(epoch_features[0]) if epoch_features else 0
            fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features), squeeze=False)
            fig.suptitle(f'Temperature Features Over Epochs - {stream}')
            title = f'Temperature Features Over Epochs - {stream}'

            for i, feature_name in enumerate(epoch_features[0].keys()):
                feature_values = [epoch_feature[feature_name] for epoch_feature in epoch_features]

                ax = axes[i, 0]
                ax.plot(feature_values, label=feature_name)
                ax.set_title(f'{feature_name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.legend(loc='upper left')

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles

    def plot_features_over_time(self, all_features, epoch_duration=5):
        figs = []
        titles = []
        for stream, epoch_features in all_features.items():
            start_time = pd.to_datetime(self.dataset[stream]['epochs'][0].index[0], unit='s')

            num_features = len(epoch_features[0]) if epoch_features else 0
            fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features), squeeze=False)
            fig.suptitle(f'Temperature Features Over Time - {stream}')
            title = f'Temperature Features Over Time - {stream}'

            for i, feature_name in enumerate(epoch_features[0].keys()):
                times_vals = [
                    (start_time + pd.to_timedelta(idx * epoch_duration, unit='s'), epoch_feature[feature_name])
                    for idx, epoch_feature in enumerate(epoch_features)]

                times, vals = zip(*times_vals)

                ax = axes[i, 0]
                ax.plot(times, vals, label=feature_name)
                ax.set_title(f'{feature_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend(loc='upper left')

                for label in ax.get_xticklabels():
                    label.set_rotation(15)
                    label.set_ha('right')

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles



