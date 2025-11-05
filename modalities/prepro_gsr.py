"""
Galvanic Skin Response (GSR) Preprocessing Module

This module provides preprocessing capabilities for GSR/EDA signals including
low-pass filtering, normalization, downsampling, and segmentation.

Galvanic Skin Response (GSR) / Electrodermal Activity (EDA) measures skin conductance changes, used for:
- Stress and arousal level monitoring
- Emotional response detection
- Anxiety assessment
- Cognitive load measurement
- Sympathetic nervous system activity tracking
- Lie detection and psychophysiological research

Classes:
    PreProGSR: Main preprocessing class for GSR/EDA signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt


class PreProGSR:
    """
    Preprocessing class for Galvanic Skin Response (GSR) data.

    This class provides a complete preprocessing pipeline for GSR signals including:
    - Low-pass filtering (removes high-frequency noise, cutoff ~1 Hz)
    - Normalization (min-max scaling to 0-1 range)
    - Downsampling
    - Segmentation into epochs

    Attributes:
        dataset (dict): Dictionary containing GSR datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets
        sfreq (float): Current stream sampling frequency
        data (pd.DataFrame): Current stream data being processed

    Example:
        >>> gsr_data = {'stream_1': {'data': gsr_df, 'sfreq': 4}}
        >>> prepro = PreProGSR(gsr_data)
        >>> processed = prepro.preprocess_gsr_data(epoch_length=30)
    """

    def __init__(self, dataset):
        """
        Initialize the GSR preprocessing object.

        Args:
            dataset (dict): Dictionary of GSR datasets where each entry contains:
                - 'data': pandas DataFrame with GSR signal data (skin conductance in µS)
                - 'sfreq': Sampling frequency in Hz

        Note:
            GSR typically sampled at low rates (4-32 Hz) as it's a slow-changing signal
        """
        self.dataset = dataset
        self.sfreq = None
        self.data = None
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

    def downsample_data(self, data, original_sfreq, target_sfreq):
        if original_sfreq <= target_sfreq:
            return data
        else:
            factor = int(original_sfreq / target_sfreq)
            downsampled_data = data.iloc[::factor, :]
            new_index = data.index[::factor]
            downsampled_data.index = new_index
            return downsampled_data

    def lowpass_filter(self, data, cutoff, order=5):
        """
        Apply low-pass filter to GSR data to remove high-frequency noise.

        Args:
            data (pd.DataFrame or pd.Series): GSR signal data
            cutoff (float): Cutoff frequency in Hz (typically ~1 Hz for GSR)
            order (int): Filter order (default: 5)

        Returns:
            array-like: Filtered GSR signal

        Note:
            - GSR is a slow-changing signal (< 1 Hz typically)
            - Cutoff of 1 Hz removes muscle artifacts and high-frequency noise
            - Returns original data if filtering fails
        """
        # Ensure data is a 1D array or Series
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]

        # Ensure data is long enough for the chosen filter order
        min_len = order * 3  # heuristic
        if len(data) < min_len:
            print(f"Data length ({len(data)}) is too short for filtering. Required minimum length is {min_len}.")
            return data  # Return the original data or handle as needed

        nyq = 0.5 * self.sfreq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        try:
            filtered_data = filtfilt(b, a, data, padtype='odd', padlen=order * 3)
        except Exception as e:
            print(f"Error during filtering GSR Data: {e}")
            return data  # Return the original data in case of error

        return filtered_data

    def normalize(self, data):
        """
        Normalize GSR data using min-max scaling to [0, 1] range.

        Args:
            data (array-like): GSR data to normalize

        Returns:
            array-like: Normalized data scaled to [0, 1]

        Note:
            Formula: (x - min) / (max - min)
            Useful for comparing GSR responses across different participants
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def preprocess(self):
        self.data = self.downsample_data(self.data, self.sfreq, self.min_sfreq)
        filtered_data = self.lowpass_filter(self.data, cutoff=1)
        preprocessed_data = self.normalize(filtered_data)

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

    def preprocess_gsr_data(self, epoch_length=5):
        """
        Complete GSR preprocessing pipeline.

        Applies the full preprocessing workflow:
        1. Downsampling to minimum sampling rate
        2. Low-pass filtering (cutoff 1 Hz)
        3. Normalization (min-max scaling to [0, 1])
        4. Segmentation into epochs

        Args:
            epoch_length (float): Length of each epoch in seconds (default: 5)

        Returns:
            dict: Processed data for each stream containing:
                - 'data': Filtered, normalized GSR data
                - 'epochs': List of segmented DataFrames
                - 'sfreq': Final sampling frequency

        Example:
            >>> prepro = PreProGSR(gsr_dataset)
            >>> processed = prepro.preprocess_gsr_data(epoch_length=30)
            >>> for stream_id, stream_data in processed.items():
            ...     print(f"{stream_id}: {len(stream_data['epochs'])} epochs")

        Note:
            Longer epochs (30-60s) recommended for GSR as it's a slow-changing signal
            Useful for stress and arousal analysis
        """
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            self.data = data_info['data']
            self.sfreq = data_info['sfreq']

            preprocessed_data = self.preprocess()

            # Ensure preprocessed_data is a DataFrame
            if isinstance(preprocessed_data, np.ndarray):
                self.data = pd.DataFrame(preprocessed_data, index=self.data.index)

            epochs = self.segment_data(segment_length=epoch_length)

            processed_data[stream_id] = {'data': self.data, 'epochs': epochs, 'sfreq': self.min_sfreq}

        return processed_data

    def plot_gsr_data(self, dataset):
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']

            # Check if data is in the expected format
            if not isinstance(data, pd.DataFrame) or data.empty:
                print(f"Invalid or empty data for stream {stream}. Skipping...")
                continue

            num_channels = data.shape[1]

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels), squeeze=False)
            fig.suptitle(f'Galvanic Skin Resistance Data ({stream})')

            for i in range(num_channels):
                axes[i, 0].plot(data.index, data.iloc[:, i], label='GSR')
                axes[i, 0].set_title(f'Channel GSR-{i + 1}')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel(f'GSR (µV)')
                axes[i, 0].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(f'GSR Data ({stream})')
        return figs, titles
