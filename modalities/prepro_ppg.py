"""
PPG (Photoplethysmography) Preprocessing Module

This module provides preprocessing capabilities for PPG signals including
filtering, noise removal, normalization, downsampling, and segmentation.

PPG signals measure blood volume changes in tissue, commonly used for:
- Heart rate monitoring
- Heart rate variability (HRV) analysis
- Pulse waveform analysis
- Cardiovascular health assessment

Classes:
    PreProPPG: Main preprocessing class for PPG signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt


class PreProPPG:
    """
    Preprocessing class for PPG (Photoplethysmography) data.

    This class provides a complete preprocessing pipeline for PPG signals including:
    - Bandpass filtering (noise removal)
    - Downsampling
    - Normalization (z-score)
    - Segmentation into epochs

    Attributes:
        dataset (dict): Dictionary containing PPG datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets

    Example:
        >>> ppg_data = {'stream_1': {'data': ppg_df, 'sfreq': 64}}
        >>> prepro = PreProPPG(ppg_data)
        >>> processed = prepro.preprocess_ppg_data(epoch_length=5)
    """

    def __init__(self, dataset):
        """
        Initialize the PPG preprocessing object.

        Args:
            dataset (dict): Dictionary of PPG datasets where each entry contains:
                - 'data': pandas DataFrame with PPG signals
                - 'sfreq': Sampling frequency in Hz

        Raises:
            ValueError: If dataset is empty or invalid
        """
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

    def butter_bandpass_filter(self, data, lowcut, highcut, sfreq, order=3):
        """
        Apply Butterworth bandpass filter to PPG data.

        Uses scipy's filtfilt for zero-phase filtering to avoid signal distortion.

        Args:
            data (array-like): PPG signal data to filter
            lowcut (float): Lower cutoff frequency in Hz
            highcut (float): Upper cutoff frequency in Hz
            sfreq (float): Sampling frequency in Hz
            order (int): Filter order (default: 3)

        Returns:
            array-like: Filtered signal

        Note:
            Uses Gustafsson's method for edge handling to minimize artifacts
        """
        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, method="gust")
        return y

    def remove_noise(self, data, sfreq):
        """
        Remove noise from PPG data using bandpass filtering.

        Performs comprehensive noise removal including:
        1. Converting data to numeric
        2. Handling inf/NaN values with forward/backward fill
        3. Applying bandpass filter (0.5-4.0 Hz typical for PPG)

        Args:
            data (pd.DataFrame): Raw PPG data
            sfreq (float): Sampling frequency in Hz

        Returns:
            pd.DataFrame: Filtered PPG data

        Raises:
            ValueError: If data contains non-numeric columns after conversion
            ValueError: If data still contains NaN/inf after handling
            ValueError: If normalized cutoff frequencies are invalid

        Note:
            Typical PPG frequency range is 0.5-4.0 Hz (30-240 BPM)
        """
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("PPG-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_ffill_bfill = data_numeric.ffill().bfill()

        # Check if NaN or inf values are still present
        if data_ffill_bfill.isnull().values.any() or np.isinf(data_ffill_bfill.values).any():
            raise ValueError("PPG-Data contains NaN or inf values after handling")

        try:
            # Ensure normalized cutoff frequency is within valid range
            normalized_low_cutoff = max(min(0.5 / (0.5 * sfreq), 0.5), 0)
            normalized_high_cutoff = max(min(4.0 / (0.5 * sfreq), 0.5), 0)

            # Check if normalized cutoff frequencies are valid
            if not 0 < normalized_low_cutoff < normalized_high_cutoff < 1:
                raise ValueError(
                    f"PPG-Invalid normalized cutoff frequencies: {normalized_low_cutoff}, {normalized_high_cutoff}")

            # Apply the bandpass filter
            filtered_data = data_ffill_bfill.apply(
                lambda x: self.butter_bandpass_filter(x, normalized_low_cutoff, normalized_high_cutoff, sfreq))
        except Exception as e:
            print(f"PPG-Error during filtering: {e}")
            return data_ffill_bfill

        return filtered_data

    def normalize_data(self, data):
        """
        Normalize PPG data using z-score standardization.

        Transforms data to have zero mean and unit variance.

        Args:
            data (pd.DataFrame): PPG data to normalize

        Returns:
            pd.DataFrame: Normalized data (mean=0, std=1)

        Note:
            Formula: (x - mean) / std
        """
        return (data - data.mean()) / data.std()

    def downsample_data(self, data, original_sfreq, target_sfreq):
        """
        Downsample PPG data to a lower sampling frequency.

        Reduces data rate by selecting every Nth sample where N = original_sfreq / target_sfreq.

        Args:
            data (pd.DataFrame): Original PPG data
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
            return data
        else:
            factor = int(original_sfreq / target_sfreq)
            downsampled_data = data.iloc[::factor, :]
            # Adjust the index for the downsampled data
            new_index = data.index[::factor]
            downsampled_data.index = new_index
            return downsampled_data

    def segment_data(self, data, segment_length, sfreq):
        """
        Segment PPG data into fixed-length epochs.

        Divides continuous PPG data into non-overlapping windows of specified length.

        Args:
            data (pd.DataFrame): Continuous PPG data
            segment_length (float): Length of each segment in seconds
            sfreq (float): Sampling frequency in Hz

        Returns:
            list: List of DataFrame segments, each of length segment_length

        Note:
            - Only returns complete segments (partial segments at end are discarded)
            - Useful for windowed analysis like HRV calculation

        Example:
            >>> segments = prepro.segment_data(data, segment_length=60, sfreq=64)
            >>> print(f"Created {len(segments)} one-minute segments")
        """
        num_samples_per_segment = int(segment_length * sfreq)
        segments = [
            data.iloc[i:i + num_samples_per_segment]
            for i in range(0, len(data), num_samples_per_segment)
            if len(data.iloc[i:i + num_samples_per_segment]) == num_samples_per_segment
        ]
        return segments

    def preprocess_ppg_data(self, epoch_length=5):
        """
        Complete PPG preprocessing pipeline.

        Applies the full preprocessing workflow:
        1. Noise removal (bandpass filtering)
        2. Downsampling to minimum sampling rate
        3. Normalization (z-score)
        4. Segmentation into epochs

        Args:
            epoch_length (float): Length of each epoch in seconds (default: 5)

        Returns:
            dict: Processed data for each stream containing:
                - 'data': Normalized continuous PPG data
                - 'epochs': List of segmented DataFrames
                - 'sfreq': Final sampling frequency

        Example:
            >>> prepro = PreProPPG(ppg_dataset)
            >>> processed = prepro.preprocess_ppg_data(epoch_length=10)
            >>> for stream_id, stream_data in processed.items():
            ...     print(f"{stream_id}: {len(stream_data['epochs'])} epochs")

        Note:
            All streams are downsampled to the minimum sampling frequency across streams
        """
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            data, sfreq = data_info['data'], data_info['sfreq']

            # Check raw data
            # print(f"Raw data for {stream_id}:", data.head())

            # Filtering and normalization
            filtered_data = self.remove_noise(data, sfreq)
            # print(f"Filtered data for {stream_id}:", filtered_data.head())

            # Downsampling
            downsampled_data = self.downsample_data(filtered_data, sfreq, self.min_sfreq)
            # print(f"Downsampled data for {stream_id}:", downsampled_data.head())

            normalized_data = self.normalize_data(downsampled_data)
            # print(f"Normalized data for {stream_id}:", normalized_data.head())

            # Segmentation
            epochs = self.segment_data(normalized_data, epoch_length, self.min_sfreq)

            processed_data[stream_id] = {
                'data': normalized_data,
                'epochs': epochs,
                'sfreq': self.min_sfreq
            }

        return processed_data

    def plot_ppg_data(self, dataset):
        """
        Generate multi-channel time-series plots of PPG data.

        Creates publication-quality plots showing all PPG channels over time
        with proper labeling and formatting.

        Args:
            dataset (dict): Dataset to plot containing 'data' and metadata

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Creates one subplot per PPG channel
            - Automatically handles multiple streams
            - Returns figures for saving or further customization

        Example:
            >>> figs, titles = prepro.plot_ppg_data(processed_data)
            >>> for fig, title in zip(figs, titles):
            ...     fig.savefig(f"{title}.png", dpi=300)
        """
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']
            num_channels = data.shape[1]

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
            fig.suptitle(f'PPG Data ({stream})')
            title = f'PPG Data ({stream})'
            for i in range(num_channels):
                axes[i].plot(data.index, data.iloc[:, i], label=f'PPG-{i + 1}')
                axes[i].set_title(f'Channel PPG-{i + 1}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(f'PPG-{i + 1} (μV)')
                axes[i].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles
