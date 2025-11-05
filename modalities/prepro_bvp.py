"""
Blood Volume Pulse (BVP) Preprocessing Module

This module provides preprocessing capabilities for BVP signals including
bandpass filtering, noise removal, downsampling, and segmentation.

Blood Volume Pulse (BVP) signals measure blood volume changes in peripheral vessels, used for:
- Heart rate monitoring and heart rate variability (HRV) analysis
- Pulse wave analysis and arterial stiffness assessment
- Stress and autonomic nervous system activity monitoring
- Cardiac output estimation
- Vascular health assessment
- Photoplethysmography (PPG) signal processing

Classes:
    PreProBVP: Main preprocessing class for BVP signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt


class PreProBVP:
    """
    Preprocessing class for Blood Volume Pulse (BVP) data.

    This class provides a complete preprocessing pipeline for BVP signals including:
    - Bandpass filtering (noise removal in 0.5-4 Hz range for heart rate)
    - Downsampling
    - Segmentation into epochs
    - Peak detection for heart rate analysis

    Attributes:
        dataset (dict): Dictionary containing BVP datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets

    Example:
        >>> bvp_data = {'stream_1': {'data': bvp_df, 'sfreq': 64}}
        >>> prepro = PreProBVP(bvp_data)
        >>> processed = prepro.preprocess_bvp_data(epoch_length=5)
    """
    def __init__(self, dataset):
        """
        Initialize the BVP preprocessing object.

        Args:
            dataset (dict): Dictionary of BVP datasets where each entry contains:
                - 'data': pandas DataFrame with BVP signal data
                - 'sfreq': Sampling frequency in Hz

        Raises:
            ValueError: If dataset is empty or invalid
        """
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
        """
        Apply Butterworth bandpass filter to BVP data.

        Uses scipy's filtfilt for zero-phase filtering. Bandpass filters preserve
        heart rate frequencies (typically 0.5-4 Hz for 30-240 BPM).

        Args:
            data (array-like): BVP signal data to filter
            lowcut (float): Low cutoff frequency in Hz
            highcut (float): High cutoff frequency in Hz
            sfreq (float): Sampling frequency in Hz
            order (int): Filter order (default: 3)

        Returns:
            array-like: Filtered signal

        Note:
            Typical BVP frequency range is 0.5-4.0 Hz (30-240 BPM heart rate)
        """
        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, method="gust")
        return y

    def remove_noise(self, data, sfreq):
        """
        Remove noise from BVP data using bandpass filtering.

        Performs comprehensive noise removal including:
        1. Converting data to numeric
        2. Handling inf/NaN values with forward/backward fill
        3. Applying bandpass filter (0.5-4.0 Hz for heart rate)

        Args:
            data (pd.DataFrame): Raw BVP data
            sfreq (float): Sampling frequency in Hz

        Returns:
            pd.DataFrame: Filtered BVP data

        Raises:
            ValueError: If data contains non-numeric columns after conversion
            ValueError: If data still contains NaN/inf after handling
            ValueError: If normalized cutoff frequency is invalid

        Note:
            Bandpass filter preserves heart rate range (30-240 BPM)
            Removes DC offset and high-frequency noise
        """
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("BVP-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_ffill_bfill = data_numeric.ffill().bfill()

        # Check if NaN or inf values are still present
        if data_ffill_bfill.isnull().values.any() or np.isinf(data_ffill_bfill.values).any():
            raise ValueError("BVP-Data contains NaN or inf values after handling")

        try:
            # Dynamically determine the low pass cutoff frequency
            normalized_cutoff = min(0.4 * (0.5 * sfreq), 0.5)
            if not 0 < normalized_cutoff < 1:
                raise ValueError(f"BVP-Invalid normalized cutoff frequency: {normalized_cutoff}")

            # Apply the bandpass filter
            filtered_data = data_ffill_bfill.apply(
                lambda x: self.butter_bandpass_filter(x, normalized_cutoff, 4, sfreq))
        except Exception as e:
            print(f"BVP-Error during filtering: {e}")
            return data_ffill_bfill  # Return NaN-handled data if filtering fails

        return filtered_data

    def downsample_data(self, data, original_sfreq, target_sfreq):
        """
        Downsample BVP data to a lower sampling frequency.

        Reduces data rate by selecting every Nth sample where N = original_sfreq / target_sfreq.

        Args:
            data (pd.DataFrame): Original BVP data
            original_sfreq (float): Original sampling frequency in Hz
            target_sfreq (float): Target sampling frequency in Hz

        Returns:
            pd.DataFrame: Downsampled data with adjusted timestamps

        Note:
            - Only downsamples if original_sfreq > target_sfreq
            - Returns original data if downsampling not needed
            - Preserves DataFrame index (timestamps)
        """
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
        """
        Segment BVP data into fixed-length epochs using time-based windowing.

        Divides continuous BVP data into non-overlapping windows of specified length
        using timestamp-based slicing for accurate temporal segmentation.

        Args:
            data (pd.DataFrame): Continuous BVP data with DatetimeIndex
            segment_length (float): Length of each segment in seconds
            sfreq (float): Sampling frequency in Hz

        Returns:
            list: List of DataFrame segments, each of length segment_length

        Note:
            - Uses time-based windowing for accurate segmentation
            - Empty segments are excluded
            - Useful for heart rate variability analysis

        Example:
            >>> segments = prepro.segment_data(data, segment_length=60, sfreq=64)
            >>> print(f"Created {len(segments)} 60-second segments for HRV")
        """
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
        """
        Complete BVP preprocessing pipeline.

        Applies the full preprocessing workflow:
        1. Noise removal (bandpass filtering 0.5-4 Hz)
        2. Downsampling to minimum sampling rate
        3. Segmentation into epochs

        Args:
            epoch_length (float): Length of each epoch in seconds (default: 5)

        Returns:
            dict: Processed data for each stream containing:
                - 'data': Filtered and downsampled continuous BVP data
                - 'epochs': List of segmented DataFrames
                - 'sfreq': Final sampling frequency

        Example:
            >>> prepro = PreProBVP(bvp_dataset)
            >>> processed = prepro.preprocess_bvp_data(epoch_length=60)
            >>> for stream_id, stream_data in processed.items():
            ...     print(f"{stream_id}: {len(stream_data['epochs'])} epochs")

        Note:
            All streams are downsampled to the minimum sampling frequency across streams
            Longer epochs (30-60s) are recommended for HRV analysis
        """
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
        """
        Plot BVP signal for a single epoch with detected peaks.

        Creates visualization showing BVP waveform with peak detection,
        useful for heart rate and pulse analysis.

        Args:
            epoch (pd.DataFrame): Single epoch with UNIX timestamp index and BVP signal

        Note:
            - Converts UNIX timestamps to datetime for visualization
            - Detects peaks using scipy.signal.find_peaks
            - Peaks represent systolic points in cardiac cycle
            - Distance parameter (60) adjusts peak detection sensitivity

        Example:
            >>> prepro.plot_bvp_per_epoch(epochs[0])
        """
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
        """
        Generate time-series plots of BVP data.

        Creates publication-quality plots showing BVP waveforms over time
        with proper labeling. Handles multiple data formats (DataFrame, Series, ndarray).

        Args:
            dataset (dict): Dataset to plot containing 'data' and metadata

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Automatically detects and converts data format
            - Supports single and multi-channel BVP data
            - Returns figures for saving or further customization

        Example:
            >>> figs, titles = prepro.plot_bvp_data(processed_data)
            >>> for fig, title in zip(figs, titles):
            ...     fig.savefig(f"{title}.png", dpi=300)
        """
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

