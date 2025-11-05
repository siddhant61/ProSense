"""
Gyroscope (GYRO) Feature Extraction Module

This module provides comprehensive feature extraction capabilities for 3-axis gyroscope signals
including time-domain, frequency-domain, angular velocity, orientation tracking, and spectral features.

Gyroscope signals measure angular velocity (rate of rotation) in three axes (X, Y, Z), typically
in units of degrees/second (°/s) or radians/second (rad/s). These signals capture rotational movements
and orientation changes.

Gyroscope features are useful for:
- Rotation and orientation tracking
- Gait analysis and balance assessment
- Fall detection (sudden rotational changes)
- Activity recognition (turning, twisting movements)
- Gesture recognition
- Head movement tracking (for VR/AR applications)
- Tremor detection (Parkinson's disease)
- Sports performance analysis (golf swing, tennis serve)

Signal characteristics:
- Static: Gyroscope reads ~0 when stationary (no rotation)
- Dynamic: Non-zero values indicate rotation rate
- Angular velocity magnitude: Overall rotation intensity
- Zero-crossing rate: Frequency of direction changes

Classes:
    FeatExGYRO: Main feature extraction class for 3-axis gyroscope signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExGYRO:
    """
    Feature extraction class for 3-axis Gyroscope (GYRO) data.

    Extracts time-domain, frequency-domain, angular velocity, orientation,
    and spectral features from preprocessed gyroscope epochs for rotation
    and movement analysis.

    Attributes:
        dataset (dict): Preprocessed gyroscope dataset with epochs

    Example:
        >>> featex = FeatExGYRO(preprocessed_gyro_data)
        >>> features = featex.extract_features()
        >>> print(features['stream_1']['angular_velocity'][0])
    """
    def __init__(self, dataset):
        """
        Initialize GYRO feature extractor.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {stream_id: {'epochs': [epoch_dataframes], 'sfreq': float}}
        """
        self.dataset = dataset

    def extract_time_domain_features(self, epochs):
        """
        Extract statistical time-domain features from each gyroscope axis.

        Args:
            epochs (list): List of gyroscope epoch DataFrames (3 columns: X, Y, Z)

        Returns:
            dict: Features for each epoch and axis

        Features per axis:
            - mean: Average angular velocity (bias/drift)
            - std: Rotation variability (movement intensity)
            - min/max: Range of rotational motion
            - skew: Asymmetry of rotation
            - kurtosis: Peakedness (sudden vs gradual turns)
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['x', 'y', 'z']):
                channel_data = epoch.iloc[:, j]
                features[i][channel] = {
                    'mean': np.mean(channel_data),
                    'std': np.std(channel_data),
                    'min': np.min(channel_data),
                    'max': np.max(channel_data),
                    'skew': skew(channel_data),
                    'kurtosis': kurtosis(channel_data)
                }
        return features

    def extract_frequency_domain_features(self, epochs):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['x', 'y', 'z']):
                fft_values = rfft(epoch.iloc[:, j].to_numpy())
                features[i][channel] = {
                    'psd': np.mean(np.abs(fft_values)**2)
                }
        return features

    def calculate_angular_velocity_magnitude(self, epochs):
        """
        Calculate overall angular velocity magnitude across all axes.

        Angular velocity magnitude = sqrt(x² + y² + z²), representing total
        rotation intensity regardless of direction.

        Args:
            epochs (list): List of gyroscope epoch DataFrames

        Returns:
            dict: Angular velocity magnitude and rate of change per epoch

        Note:
            - Higher magnitude = more intense rotational movement
            - Useful for detecting turning, spinning, or head rotation
        """
        features = {}
        for i, epoch in enumerate(epochs):
            angular_velocity = np.sqrt((epoch ** 2).sum(axis=1))
            avg_velocity = np.mean(angular_velocity)
            rate_of_change = np.mean(np.diff(angular_velocity))
            features[i] = {
                'angular_velocity': {'avg_angular_velocity': avg_velocity},
                'rate_of_change': {'rate_of_change_angular_velocity': rate_of_change}
            }
        return features

    def calculate_zero_crossing_rate(self, epochs):
        """
        Calculate zero-crossing rate - frequency of direction reversals.

        Args:
            epochs (list): List of gyroscope epoch DataFrames

        Returns:
            dict: Zero-crossing count per epoch and axis

        Note:
            - High zero-crossing rate = frequent direction changes
            - Indicates oscillatory or tremor-like movements
            - Useful for detecting Parkinson's tremor
        """
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['x', 'y', 'z']):
                channel_data = epoch.iloc[:, j]
                zero_crossings = ((channel_data[:-1] * channel_data[1:]) < 0).sum()
                features[i][channel] = {'zero_crossing_rate': zero_crossings}
        return features

    def extract_spectral_energy(self, epochs):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['x', 'y', 'z']):
                fft_values = rfft(epoch.iloc[:, j].to_numpy())
                psd = np.mean(np.abs(fft_values) ** 2)
                features[i][channel] = {'spectral_energy': psd}
        return features

    def infer_sensor_location(self, stream_id):
        if 'muse' in stream_id.lower():
            return 'head'
        else:
            return 'head' # add logic for other locations later

    def location_specific_feature(self, epochs, location):
        if location == 'head':
            return self.track_orientation_angles(epochs)
        return None

    def track_orientation_angles(self, epochs):
        features = {}
        for i, epoch in enumerate(epochs):
            # Cumulative sum of angular velocities gives orientation changes
            orientation_change = np.cumsum(epoch, axis=0)
            overall_orientation = orientation_change.iloc[-1]

            # Calculate rate of change of orientation
            # This is the average of the absolute first derivatives (differences) of the orientation angles
            orientation_diff = np.diff(orientation_change, axis=0)
            rate_of_change = np.mean(np.abs(orientation_diff), axis=0)

            features[i] = {
                'orientation_change': {'overall_orientation_change': overall_orientation},
                'rate_of_change': {'rate_of_change_orientation': rate_of_change}
            }
        return features

    def extract_features(self):
        """
        Extract comprehensive gyroscope features from all epochs.

        Main entry point for GYRO feature extraction. Processes all streams to extract
        time-domain, frequency-domain, angular velocity, orientation, and spectral features.

        Returns:
            dict: Features for each stream with structure:
                {stream_id: {
                    'time_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'freq_features': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'angular_velocity': {epoch_idx: {...}},
                    'orientation_change': {epoch_idx: {...}} or None,
                    'spectral_energy': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}},
                    'zero_crossing_rate': {epoch_idx: {'x': {...}, 'y': {...}, 'z': {...}}}
                }}
        """
        all_features = {}
        for stream, data_info in self.dataset.items():
            epochs = data_info['epochs']
            time_features = self.extract_time_domain_features(epochs)
            freq_features = self.extract_frequency_domain_features(epochs)
            angular_velocity = self.calculate_angular_velocity_magnitude(epochs)
            spectral_energy = self.extract_spectral_energy(epochs)
            zero_crossing_rate = self.calculate_zero_crossing_rate(epochs)

            sensor_location = self.infer_sensor_location(stream)
            orientation_change = self.location_specific_feature(epochs, sensor_location)

            all_features[stream] = {
                "time_features": time_features,
                "freq_features": freq_features,
                "angular_velocity": angular_velocity,
                "orientation_change": orientation_change,
                "spectral_energy": spectral_energy,
                "zero_crossing_rate": zero_crossing_rate,
            }
        return all_features

    def plot_features(self, all_features, plot_type='ind'):
        plots = {}
        for stream, feature_data in all_features.items():
            plots[stream] = []
            for feature_type, epoch_features in feature_data.items():
                num_epochs = len(epoch_features)

                if feature_type in ["angular_velocity", "orientation_change"]:
                    # These features have a different structure
                    plots[stream].append(self.plot_special_features(stream, epoch_features, feature_type, num_epochs, plot_type))
                else:
                    # Regular feature plotting
                    plots[stream].append(self.plot_regular_features(stream, epoch_features, feature_type, num_epochs, plot_type))
        return plots
    def plot_regular_features(self, stream, epoch_features, feature_type, num_epochs, plot_type):
        figs = []
        titles = []
        if plot_type == 'all':
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
            fig.suptitle(f'{feature_type.capitalize()} over Epochs', fontsize=16)
            title = f'{feature_type.capitalize()} over Epochs'
            for j, channel in enumerate(['x', 'y', 'z']):
                ax = axes[j]
                for feature in epoch_features[0]['x'].keys():
                    channel_feature_values = [epoch_features[i][channel][feature] for i in range(num_epochs)]
                    ax.plot(range(num_epochs), channel_feature_values,
                            label=f'{channel.upper()} Channel - {feature.capitalize()}')
                ax.set_title(f'{channel.upper()} Channel')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'{feature.capitalize()} Value')
                ax.legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        else:
            num_epochs = len(epoch_features)
            # Iterate over each feature within the channel
            for feature in epoch_features[0]['x'].keys():
                fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                fig.suptitle(f'{stream} - {feature.capitalize()} over Epochs', fontsize=16)
                title = f'{stream} - {feature.capitalize()} over Epochs'
                for j, channel in enumerate(['x', 'y', 'z']):
                    channel_feature_values = [epoch_features[i][channel][feature] for i in range(num_epochs)]

                    ax = axes[j]
                    ax.plot(range(num_epochs), channel_feature_values,
                            label=f'{channel.upper()} Channel - {feature.capitalize()}')
                    ax.set_title(f'{channel.upper()} Channel')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(f'{feature.capitalize()} Value')
                    ax.legend()

                plt.tight_layout()
                figs.append(fig)
                titles.append(title)
        return figs, titles

    def plot_special_features(self, stream, epoch_features, feature_type, num_epochs, plot_type):
        figs = []
        titles = []
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # One subplot for each channel
            fig.suptitle(f'{feature_type.capitalize()} over Epochs', fontsize=16)
            title = f'{feature_type.capitalize()} over Epochs'
            for j, channel in enumerate(list(epoch_features[0].keys())):
                ax = axes[j]

                for feature in list(epoch_features[0][channel].keys()):
                    if feature_type == "orientation_change":
                        channel_feature_values = [epoch_features[i][channel][feature] for i in range(num_epochs)]
                        ax.plot(range(num_epochs), channel_feature_values,
                                label=['x','y','z'])
                    else:
                        channel_feature_values = [epoch_features[i][channel][feature] for i in range(num_epochs)]
                        ax.plot(range(num_epochs), channel_feature_values,
                                label=f'{feature.capitalize()}')
                ax.set_title(f'{channel.upper()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'{feature.capitalize()}')
                ax.legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        else:
            num_epochs = len(epoch_features)

            if feature_type == "orientation_change":

                for j, feature in enumerate(['orientation_change', 'rate_of_change']):
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                    fig.suptitle(f'{stream} - {feature} over Epochs', fontsize=16)
                    title = f'{stream} - {feature} over Epochs'
                    for k, channel in enumerate(['x', 'y', 'z']):
                        # Get feature values for the specific channel only
                        channel_feature_values = [epoch_features[i][feature][list(epoch_features[i][feature].keys())[0]][k] for i in range(num_epochs)]
                        ax = axes[k]
                        ax.plot(range(num_epochs), channel_feature_values,
                                label=f'{channel.upper()} Channel - {feature.capitalize()}')
                        ax.set_title(f'{channel.upper()} Channel')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel(f'{feature.capitalize()} Value')
                        ax.legend()

                    plt.tight_layout()
                    figs.append(fig)
                    titles.append(title)

            else:
                num_epochs = len(epoch_features)

                fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # One subplot for each channel
                fig.suptitle(f'{stream} - angular_velocity over Epochs', fontsize=16)
                title = f'{stream} - angular_velocity over Epochs'
                for j, channel in enumerate(['avg_angular_velocity', 'rate_of_change_angular_velocity']):
                    channel_feature_values = [epoch_features[i][list(epoch_features[0].keys())[j]][channel] for i in range(num_epochs)]
                    feature = channel
                    ax = axes[j]
                    ax.plot(range(num_epochs), channel_feature_values,
                            label=f'{channel.upper()}')
                    ax.set_title(f'{channel.upper()}')
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

        for stream, feature_categories in all_features.items():
            for category, category_features in feature_categories.items():
                for feature, feature_data in category_features.items():
                    # Assuming 'feature_data' is a list of dictionaries for each epoch
                    num_epochs = len(feature_data)

                    # Initialize lists for time points and feature values
                    time_points = pd.date_range(start=0, periods=num_epochs, freq=pd.DateOffset(seconds=epoch_duration))
                    feature_values = []

                    # Extract feature values from each epoch
                    for epoch in feature_data:
                        if isinstance(epoch, dict) and feature in epoch:
                            feature_values.append(epoch[feature])

                    # Skip if no data available for this feature
                    if not feature_values:
                        continue

                    # Plotting
                    fig, ax = plt.subplots(figsize=(12, 6))
                    title = f'{stream} - {category} - {feature} over Time'
                    fig.suptitle(title, fontsize=16)

                    ax.plot(time_points, feature_values, marker='o', label=f'{feature}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{feature} Value')
                    ax.legend()

                    plt.tight_layout()
                    figs.append(fig)
                    titles.append(title)

        return figs, titles








