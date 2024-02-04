import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class PreProTEMP:

    def __init__(self, dataset):
        self.dataset = dataset
        self.sfreq = None
        self.data= None
        # Determine the minimum sampling frequency across all streams
        sfreqs = [data_info['sfreq'] for data_info in dataset.values()]
        if sfreqs:
            self.min_sfreq = min(sfreqs)
        else:
            self.min_sfreq = min(self.calculate_sampling_frequency(data['data']) for data in dataset.values())
            if not self.min_sfreq:
                self.min_sfreq = 2.5

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

    def downsample_data(self, data, original_sfreq, target_sfreq):
        if original_sfreq <= target_sfreq:
            return data
        else:
            factor = int(original_sfreq / target_sfreq)
            downsampled_data = data.iloc[::factor, :]
            # Adjust the index for the downsampled data
            new_index = data.index[::factor]
            downsampled_data.index = new_index
            return downsampled_data

    def smooth_data(self, window_size=5):
        # Assuming self.data is a DataFrame and you are interested in the first column
        # You can change this to select a different column as needed
        if isinstance(self.data, pd.DataFrame):
            data_series = self.data.iloc[:, 0]
        else:
            data_series = self.data

        return np.convolve(data_series, np.ones((window_size,)) / window_size, mode='valid')

    def baseline_correction(self, data):
        linear_drift = np.linspace(0, np.mean(data), len(data))
        return data - linear_drift

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def preprocess(self):
        self.data = self.downsample_data(self.data, self.sfreq, self.min_sfreq)
        smoothed_data = self.smooth_data()
        corrected_data = self.baseline_correction(smoothed_data)
        preprocessed_data = self.normalize(corrected_data)

        if len(preprocessed_data) != len(self.data.index):
            # Option 1: Padding (example using zero padding)
            padding_length = len(self.data.index) - len(preprocessed_data)
            preprocessed_data = np.pad(preprocessed_data, (0, padding_length), mode='constant')

        # Ensure preprocessed_data is a DataFrame
        if isinstance(preprocessed_data, np.ndarray):
            preprocessed_data = pd.DataFrame(preprocessed_data, index=self.data.index)

        return preprocessed_data

    def segment_data(self, segment_length):
        samples_per_segment = int(segment_length * self.min_sfreq)
        segments = []
        num_segments = int(np.ceil(len(self.data) / samples_per_segment))

        for i in range(num_segments):
            # Use Timedelta for timestamp arithmetic
            start_time = self.data.index[0] + pd.Timedelta(seconds=i * segment_length)
            end_time = start_time + pd.Timedelta(seconds=segment_length)

            segment = self.data[(self.data.index >= start_time) & (self.data.index < end_time)]
            if not segment.empty:
                segments.append(segment)

        return segments

    def preprocess_temp_data(self, epoch_length=5):
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            self.data = data_info['data']
            self.sfreq = data_info['sfreq']

            preprocessed_data = self.preprocess()

            if isinstance(preprocessed_data, np.ndarray):
                self.data = pd.DataFrame(preprocessed_data, index=self.data.index)

            epochs = self.segment_data(segment_length=epoch_length)

            processed_data[stream_id] = {'data': self.data, 'epochs': epochs, 'sfreq': self.min_sfreq}

        return processed_data

    def plot_temp_data(self, dataset):
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']

            # Check if data is a numpy array and convert it to DataFrame
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=['Temperature'])  # Assuming single column data
                num_channels = 1
            elif isinstance(data, pd.DataFrame):
                num_channels = data.shape[1]
            elif isinstance(data, pd.Series):
                data = data.to_frame()  # Convert Series to DataFrame
                num_channels = 1
            else:
                print(f"Invalid data format for stream {stream}. Skipping...")
                continue

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels), squeeze=False)
            fig.suptitle(f'Temperature Data ({stream})')
            title = f'Temperature Data ({stream})'

            for i in range(num_channels):
                axes[i, 0].plot(data.index, data.iloc[:, i], label='Temperature')
                axes[i, 0].set_title(f'Temperature Channel {i + 1}' if num_channels > 1 else 'Temperature')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('Temperature (C)')
                axes[i, 0].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles



