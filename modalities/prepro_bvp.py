import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

class PreProBVP:
    def __init__(self, dataset):
        self.dataset = dataset
        # Determine the minimum sampling frequency across all data streams
        sfreqs = [data_info['sfreq'] for data_info in dataset.values()]
        if sfreqs is None:
            self.min_sfreq = min(self.calculate_sampling_frequency(data['data']) for data in dataset.values())
        else:
            self.min_sfreq = min(sfreqs)

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

    def butter_bandpass_filter(self, data, lowcut, highcut, sfreq, order=3):
        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, method="gust")
        return y

    def remove_noise(self, data, sfreq):
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("PPG-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_ffill_bfill = data_numeric.fillna(method='ffill').fillna(method='bfill')

        # Check if NaN or inf values are still present
        if data_ffill_bfill.isnull().values.any() or np.isinf(data_ffill_bfill.values).any():
            raise ValueError("PPG-Data contains NaN or inf values after handling")

        try:
            # Dynamically determine the low pass cutoff frequency
            normalized_cutoff = min(0.4 * (0.5 * sfreq), 0.5)
            if not 0 < normalized_cutoff < 1:
                raise ValueError(f"PPG-Invalid normalized cutoff frequency: {normalized_cutoff}")

            # Apply the lowpass filter
            filtered_data = data_ffill_bfill.apply(
                lambda x: self.butter_bandpass_filter(x, normalized_cutoff, 4, sfreq))
        except Exception as e:
            print(f"PPG-Error during filtering: {e}")
            return data_ffill_bfill  # Return NaN-handled data if filtering fails

        return filtered_data

    def downsample_data(self, data, original_sfreq, target_sfreq):
        if original_sfreq <= target_sfreq:
            # No downsampling needed
            return data
        else:
            # Calculate downsampling factor
            factor = int(original_sfreq / target_sfreq)

            # Downsample by selecting every 'factor'-th sample
            downsampled_data = data.iloc[::factor]

            # Adjust the index for the downsampled data
            new_index = data.index[::factor]
            downsampled_data.index = new_index

            return downsampled_data

    def segment_data(self, data, segment_length, sfreq):
        # Convert segment length from seconds to number of samples
        samples_per_segment = segment_length * sfreq
        segments = []

        # Calculate the number of segments
        num_segments = int(np.ceil(len(data) / samples_per_segment))

        for i in range(num_segments):
            # Calculate start and end times for each segment using Timedelta
            start_time = data.index[0] + pd.Timedelta(seconds=i * segment_length)
            end_time = start_time + pd.Timedelta(seconds=segment_length)

            # Select data within the time window
            segment = data[(data.index >= start_time) & (data.index < end_time)]

            # Append the segment if it's not empty
            if not segment.empty:
                segments.append(segment)

        return segments


    def preprocess_bvp_data(self, epoch_length=5):
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            data = data_info['data']
            sfreq = data_info['sfreq']

            # Filter the data
            filtered_data = self.remove_noise(data, sfreq)

            # Downsample if necessary
            downsampled_data = self.downsample_data(filtered_data, sfreq, self.min_sfreq)

            # Segment the data
            epochs = self.segment_data(downsampled_data, segment_length=epoch_length, sfreq=self.min_sfreq)

            # for epoch in epochs:
            #     self.plot_bvp_per_epoch(epoch)

            processed_data[stream_id] = {'data': downsampled_data,'epochs': epochs, 'sfreq': self.min_sfreq}

        return processed_data

    # def plot_bvp_data(self, dataset):
    #     figs = []
    #     titles = []
    #     for stream, data_info in dataset.items():
    #         data = data_info['data']
    #         epochs = self.segment_data(data, segment_length=5, sfreq=self.min_sfreq)
    #
    #         # Determine the number of channels from the first epoch
    #         num_channels = epochs[0].shape[1] if isinstance(epochs[0], pd.DataFrame) else 1
    #
    #         # Create subplots for each channel
    #         fig, axes = plt.subplots(num_channels, 1, figsize=(12, num_channels * 3), squeeze=False)
    #         fig.suptitle(f'Blood Volume Pulse Data - {stream}')
    #         title = f'Blood Volume Pulse Data - {stream}'
    #
    #         # Plot the collected data for each channel
    #         for i in range(num_channels):
    #             for epoch in epochs:
    #                 # Convert UNIX timestamps to datetime objects
    #                 epoch.index = pd.to_datetime(epoch.index, unit='s')
    #
    #                 # Select the data to plot
    #                 data_to_plot = epoch.iloc[:, i] if isinstance(epoch, pd.DataFrame) else epoch
    #
    #                 # Plot the data
    #                 axes[i, 0].plot(epoch.index, data_to_plot, label=f'Epoch {epoch.index[0]}')
    #
    #             axes[i, 0].set_title(f'BVP Signal - Channel {i + 1}')
    #             axes[i, 0].set_xlabel('Time')
    #             axes[i, 0].set_ylabel('BVP (μV)')
    #             axes[i, 0].legend()
    #
    #         plt.tight_layout()
    #         figs.append(fig)
    #         titles.append(title)
    #
    #     return figs, titles

    def plot_bvp_per_epoch(self, epoch):
        # Assuming 'epoch' is a DataFrame with a UNIX timestamp index and BVP signal in the first column
        if isinstance(epoch, pd.DataFrame):
            # Convert UNIX timestamps to datetime objects
            epoch.index = pd.to_datetime(epoch.index, unit='s')
            signal = epoch.iloc[:, 0]

            # Find peaks (make sure to pass the signal values, not the DataFrame)
            peaks, _ = find_peaks(signal.values, distance=60)  # Adjust parameters as necessary

            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(epoch.index, signal, label='BVP Signal')

            # Plot the peaks, converting the peak indices to the corresponding timestamps
            peak_times = epoch.index[peaks]
            plt.plot(peak_times, signal.iloc[peaks], "x", label='Detected Peaks')

            # Update labels for clarity
            plt.title("BVP Signal with Detected Peaks")
            plt.xlabel("Time")
            plt.ylabel("BVP (μV)")
            plt.legend()


    def plot_bvp_data(self, dataset):
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']

            # Check if data is a numpy array and convert it to DataFrame
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=['BVP'])  # Assuming single column data
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
            fig.suptitle(f'BVP Data ({stream})')
            title = f'BVP Data ({stream})'

            for i in range(num_channels):
                axes[i, 0].plot(data.index, data.iloc[:, i], label='BVP')
                axes[i, 0].set_title(f'BVP Channel {i + 1}' if num_channels > 1 else 'BVP')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('BVP (μV)')
                axes[i, 0].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles

