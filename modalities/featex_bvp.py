import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExBVP:
    def __init__(self, dataset):
        self.dataset = dataset

    def extract_time_domain_features(self, epochs):
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





