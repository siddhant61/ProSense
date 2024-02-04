import numpy as np
import pandas as pd
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FeatExPPG:
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_heart_rate(self, epochs, sfreq, heart_rate_range=(40, 180)):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
                channel_data = epoch.iloc[:, j].dropna()

                # Calculate the minimum distance between peaks
                min_distance = sfreq / (heart_rate_range[1] / 60)
                min_distance = max(1, int(min_distance))

                # Improved peak detection
                peaks, _ = find_peaks(channel_data, distance=min_distance)

                # Calculate heart rate
                peak_intervals = np.diff(peaks) / sfreq
                heart_rate = np.array(60 / peak_intervals) if len(peak_intervals) > 0 else np.array([np.nan])

                # Filter out implausible heart rate values
                valid_heart_rate = heart_rate[(heart_rate > heart_rate_range[0]) & (heart_rate < heart_rate_range[1])]
                avg_heart_rate = np.nanmean(valid_heart_rate) if len(valid_heart_rate) > 0 else np.nan

                features[i][channel] = {'heart_rate': avg_heart_rate}
        return features

    def calculate_hrv(self, epochs, sfreq, heart_rate_range=(40, 180)):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
                channel_data = epoch.iloc[:, j].dropna()
                # Apply median filter for smoothing
                channel_data_smoothed = medfilt(channel_data, kernel_size=5)

                # Calculate the minimum distance between peaks
                min_distance = sfreq / (heart_rate_range[1] / 60)
                min_distance = max(1, int(min_distance))

                # Advanced peak detection (customize as needed)
                peaks, _ = find_peaks(channel_data_smoothed, distance=min_distance)

                # Calculate heart rate and validate
                peak_intervals = np.diff(peaks) / sfreq
                heart_rate = 60 / peak_intervals
                valid_heart_rate = (heart_rate > heart_rate_range[0]) & (heart_rate < heart_rate_range[1])
                valid_intervals = peak_intervals[valid_heart_rate]

                # HRV calculation (using standard deviation or alternative metrics)
                hrv = np.std(valid_intervals) if len(valid_intervals) > 1 else None
                features[i][channel] = {'hrv': hrv}
        return features

    def extract_prv(self, epochs, sfreq):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
                channel_data = epoch.iloc[:, j].dropna()

                # Improved peak detection
                peaks, _ = find_peaks(channel_data, distance=sfreq / 2, height=np.mean(channel_data))

                # Skip if too few peaks
                if len(peaks) < 2:
                    features[i][channel] = {'prv': None}
                    continue

                peak_intervals = np.diff(peaks) / sfreq
                # Remove outliers from peak intervals
                median_pi = np.median(peak_intervals)
                mad_pi = np.median(np.abs(peak_intervals - median_pi))
                valid_intervals = peak_intervals[
                    (peak_intervals > median_pi - 2 * mad_pi) & (peak_intervals < median_pi + 2 * mad_pi)]

                # Robust measure of variability
                prv = np.std(valid_intervals) if len(valid_intervals) > 1 else None
                features[i][channel] = {'prv': prv}
        return features

    def extract_amplitude(self, epochs):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
                channel_data = epoch.iloc[:, j]
                amplitude = np.max(channel_data) - np.min(channel_data)
                features[i][channel] = {'amplitude': amplitude}
        return features

    def infer_sensor_location(self, stream_id):
        if 'muse' in stream_id.lower():
            return 'head'
        # Add other location inferences here
        return 'unknown'

    def location_specific_feature(self, epochs, location):
        # Example: Blood flow feature if the sensor is on the head
        if location == 'head':
            return self.calculate_head_blood_flow(epochs)
        return None

    def calculate_head_blood_flow(self, epochs):
        features = {}
        for i, epoch in enumerate(epochs):
            features[i] = {}
            for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
                channel_data = epoch.iloc[:, j]
                # Higher amplitude might indicate increased blood flow, but this is a simplification
                peaks, _ = find_peaks(channel_data)
                troughs, _ = find_peaks(-channel_data)
                if len(peaks) > 0 and len(troughs) > 0:
                    average_peak = np.mean(channel_data.iloc[peaks])
                    average_trough = np.mean(channel_data.iloc[troughs])
                    average_amplitude = average_peak - average_trough
                else:
                    average_amplitude = np.nan

                features[i][channel] = {'head_blood_flow': average_amplitude}
        return features

    def extract_features(self):
        all_features = {}
        for stream, data_info in self.dataset.items():
            sfreq = data_info['sfreq']
            epochs = data_info['epochs']
            sensor_location = self.infer_sensor_location(stream)
            hr_features = self.calculate_heart_rate(epochs, sfreq)
            hrv_features = self.calculate_hrv(epochs, sfreq)
            prv_features = self.extract_prv(epochs, sfreq)
            amp_features = self.extract_amplitude(epochs)
            flow_features = self.location_specific_feature(epochs, sensor_location)

            # Combine all features into a single dictionary for the file
            all_features[stream] = {
                "hr_features": hr_features,
                "hrv_features": hrv_features,
                "prv_features": prv_features,
                "amp_features": amp_features,
                "flow_features": flow_features
            }

        return all_features

    def plot_features(self, all_features):
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                num_epochs = len(epoch_features)

                # Iterate over each feature within the channel
                for feature in epoch_features[0]['PPG1'].keys():
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                    fig.suptitle(f'{stream} - {feature.capitalize()} over Epochs', fontsize=16)
                    title = f'{stream} - {feature.capitalize()} over Epochs'

                    for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
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



    def plot_features_over_time(self, all_features, epoch_duration=5):
        figs = []
        titles = []
        for stream, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                # Iterate over each feature within the channel
                for feature in epoch_features[0]['PPG1'].keys():
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # One subplot for each channel
                    fig.suptitle(f'{stream} - {feature.capitalize()} over Time', fontsize=16)
                    title = f'{stream} - {feature.capitalize()} over Time'

                    for j, channel in enumerate(['PPG1', 'PPG2', 'PPG3']):
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