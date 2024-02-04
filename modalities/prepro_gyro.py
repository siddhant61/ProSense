import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

class PreProGYRO:
    def __init__(self, dataset):
        self.dataset = dataset
        # Determine the minimum sampling frequency across all streams
        sfreqs = [data_info['sfreq'] for data_info in dataset.values()]
        if sfreqs:
            self.min_sfreq = min(sfreqs)
        else:
            self.min_sfreq = min(self.calculate_sampling_frequency(data['data']) for data in dataset.values())

    def calculate_sampling_frequency(self, data):
        """
        Calculate the sampling frequency of the data.

        Parameters:
        data (pd.DataFrame): DataFrame with timestamps as index.

        Returns:
        float: Estimated sampling frequency.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")

        if len(data) < 2:
            raise ValueError("The DataFrame does not have enough data points to calculate a sampling frequency.")

        # Calculate time differences between consecutive data points
        time_diffs = data.index.to_series().diff().dropna()

        # Calculate average time difference in seconds
        avg_time_diff = time_diffs.mean().total_seconds()

        # Sampling frequency is the inverse of the average time difference
        sfreq = 1 / avg_time_diff

        return sfreq

    def butter_lowpass_filter(self, data, cutoff, sfreq, order=5):
        nyq = 0.5 * sfreq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def remove_noise(self, data, sfreq):
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("GYRO-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_ffill_bfill = data_numeric.fillna(method='ffill').fillna(method='bfill')

        # Check if NaN or inf values are still present
        if data_ffill_bfill.isnull().values.any() or np.isinf(data_ffill_bfill.values).any():
            raise ValueError("GYRO-Data contains NaN or inf values after handling")

        try:
            # Dynamically determine the low pass cutoff frequency
            normalized_cutoff = min(0.4 * (0.5 * sfreq), 0.5)
            if not 0 < normalized_cutoff < 1:
                raise ValueError(f"GYRO-Invalid normalized cutoff frequency: {normalized_cutoff}")

            # Apply the lowpass filter
            filtered_data = data_ffill_bfill.apply(lambda x: self.butter_lowpass_filter(x, normalized_cutoff, sfreq))
        except Exception as e:
            print(f"GYRO-Error during filtering: {e}")
            return data_ffill_bfill  # Return NaN-handled data if filtering fails

        return filtered_data

    def normalize_data(self, data):
        return (data - data.mean()) / data.std()

    def downsample_data(self, data, original_sfreq, target_sfreq):
        if original_sfreq <= target_sfreq:
            return data
        else:
            factor = int(original_sfreq / target_sfreq)
            downsampled_data = data.iloc[::factor, :]
            # Adjust the index for the downsampled data
            new_index = data.index[::factor]  # Assuming index is numeric and represents time
            downsampled_data.index = new_index
            return downsampled_data

    def segment_data(self, data, segment_length, sfreq):
        """Segment the data into fixed-size windows."""
        num_samples_per_segment = int(segment_length * sfreq)
        segments = [
            data.iloc[i:i + num_samples_per_segment]
            for i in range(0, len(data), num_samples_per_segment)
            if len(data.iloc[i:i + num_samples_per_segment]) == num_samples_per_segment
        ]
        return segments

    def infer_sensor_location(self, stream_id):
        if 'muse' in stream_id.lower():
            return 'head'
        elif any(char.isdigit() for char in stream_id) and not 'muse' in stream_id.lower():
            return 'wrist'
        else:
            return 'unknown'

    def preprocess_gyro_data(self, epoch_length=5):
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            data, sfreq = data_info['data'], data_info['sfreq']
            filtered_data = self.remove_noise(data, sfreq)

            downsampled_data = self.downsample_data(filtered_data, sfreq, self.min_sfreq)

            normalized_data = self.normalize_data(downsampled_data)

            # Segmentation
            epochs = self.segment_data(normalized_data, epoch_length, self.min_sfreq)

            processed_data[stream_id] = {
                'data': normalized_data,
                'epochs':epochs,
                'sfreq': self.min_sfreq}
        return processed_data

    def plot_gyro_data(self, dataset):
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']
            num_channels = data.shape[1]
            sensor_location = self.infer_sensor_location(stream)

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
            fig.suptitle(f'Gyroscope Data - {sensor_location.capitalize()} ({stream})')
            title = f'Gyroscope Data - {sensor_location.capitalize()} ({stream})'
            for i, channel in enumerate(['x', 'y', 'z']):
                axes[i].plot(data.index, data.iloc[:, i], label=f'GYRO-{channel}')
                axes[i].set_title(f'Channel GYRO-{channel}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(f'GYRO-{channel} (μV)')
                axes[i].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles
