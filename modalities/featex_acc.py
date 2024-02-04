import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExACC:
    def __init__(self, dataset):
        self.dataset = dataset

    def extract_time_domain_features(self, epochs):
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
        """Extract various features from the EEG data."""
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



