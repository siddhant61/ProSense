"""
Accelerometer (ACC) Preprocessing Module

This module provides preprocessing capabilities for accelerometer signals including
filtering, noise removal, normalization, downsampling, and segmentation.

Accelerometer signals measure acceleration forces in three axes (X, Y, Z), commonly used for:
- Activity recognition (walking, running, sitting, standing)
- Fall detection and balance assessment
- Movement quality analysis
- Gesture recognition
- Physical activity monitoring and energy expenditure estimation

Classes:
    PreProACC: Main preprocessing class for 3-axis accelerometer signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# Constants
LOW_PASS_CUTOFF = 5  # Low pass filter cutoff frequency (can be adjusted)


class PreProACC:
    """
    Preprocessing class for Accelerometer (ACC) data.

    This class provides a complete preprocessing pipeline for 3-axis accelerometer signals including:
    - Low-pass filtering (noise removal)
    - Missing value handling (interpolation)
    - Downsampling
    - Normalization (z-score)
    - Segmentation into epochs
    - Sensor location inference

    Attributes:
        dataset (dict): Dictionary containing ACC datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets

    Example:
        >>> acc_data = {'stream_1': {'data': acc_df, 'sfreq': 32}}
        >>> prepro = PreProACC(acc_data)
        >>> processed = prepro.preprocess_acc_data(epoch_length=5)
    """
    def __init__(self, dataset):
        """
        Initialize the ACC preprocessing object.

        Args:
            dataset (dict): Dictionary of ACC datasets where each entry contains:
                - 'data': pandas DataFrame with 3-axis accelerometer data (X, Y, Z)
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

    def butter_lowpass_filter(self, data, cutoff, sfreq, order=5):
        """
        Apply Butterworth low-pass filter to accelerometer data.

        Uses scipy's filtfilt for zero-phase filtering to avoid signal distortion.
        Low-pass filters remove high-frequency noise while preserving body movements.

        Args:
            data (array-like): Accelerometer signal data to filter
            cutoff (float): Cutoff frequency in Hz
            sfreq (float): Sampling frequency in Hz
            order (int): Filter order (default: 5)

        Returns:
            array-like: Filtered signal

        Note:
            Typical cutoff for human movement is 5-20 Hz
            Higher order filters have sharper cutoff but more computational cost
        """
        nyq = 0.5 * sfreq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def remove_noise(self, data, sfreq):
        """
        Remove noise from accelerometer data using low-pass filtering.

        Performs comprehensive noise removal including:
        1. Converting data to numeric
        2. Handling inf/NaN values with forward/backward fill
        3. Applying low-pass filter to remove high-frequency noise

        Args:
            data (pd.DataFrame): Raw accelerometer data (3 axes: X, Y, Z)
            sfreq (float): Sampling frequency in Hz

        Returns:
            pd.DataFrame: Filtered accelerometer data

        Raises:
            ValueError: If data contains non-numeric columns after conversion
            ValueError: If data still contains NaN/inf after handling
            ValueError: If normalized cutoff frequency is invalid

        Note:
            Low-pass filter removes vibrations and sensor noise while preserving
            human body movements (typically < 20 Hz)
        """
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("ACC-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_ffill_bfill = data_numeric.ffill().bfill()

        # Check if NaN or inf values are still present
        if data_ffill_bfill.isnull().values.any() or np.isinf(data_ffill_bfill.values).any():
            raise ValueError("ACC-Data contains NaN or inf values after handling")

        try:
            # Dynamically determine the low pass cutoff frequency
            normalized_cutoff = min(0.4 * (0.5 * sfreq), 0.5)
            if not 0 < normalized_cutoff < 1:
                raise ValueError(f"ACC-Invalid normalized cutoff frequency: {normalized_cutoff}")

            # Apply the lowpass filter
            filtered_data = data_ffill_bfill.apply(lambda x: self.butter_lowpass_filter(x, normalized_cutoff, sfreq))
        except Exception as e:
            print(f"ACC-Error during filtering: {e}")
            return data_ffill_bfill  # Return NaN-handled data if filtering fails

        return filtered_data

    def handle_missing_values(self, data):
        """
        Handle missing values using linear interpolation.

        Interpolates missing values to maintain temporal continuity in the signal.

        Args:
            data (pd.DataFrame): Accelerometer data with potential missing values

        Returns:
            pd.DataFrame: Data with interpolated values

        Note:
            Linear interpolation is suitable for short gaps in accelerometer data
        """
        # Interpolate missing values
        return data.interpolate()

    def normalize_data(self, data):
        """
        Normalize accelerometer data using z-score standardization.

        Transforms data to have zero mean and unit variance.

        Args:
            data (pd.DataFrame): Accelerometer data to normalize

        Returns:
            pd.DataFrame: Normalized data (mean=0, std=1)

        Note:
            Formula: (x - mean) / std
            Normalization makes data comparable across different sensors and locations
        """
        return (data - data.mean()) / data.std()

    def infer_sensor_location(self, stream_id):
        """
        Infer the location of the accelerometer sensor from stream ID.

        Uses heuristics to determine sensor placement on the body.

        Args:
            stream_id (str): Stream identifier string

        Returns:
            str: Inferred location ('head', 'wrist', or 'unknown')

        Note:
            - 'muse' in stream_id → head-mounted sensor
            - Numeric characters in stream_id → wrist-worn sensor
            - Otherwise → unknown location
        """
        if 'muse' in stream_id.lower():
            return 'head'
        elif any(char.isdigit() for char in stream_id):
            return 'wrist'
        else:
            return 'unknown'

    def downsample_data(self, data, original_sfreq, target_sfreq):
        """
        Downsample accelerometer data to a lower sampling frequency.

        Reduces data rate by selecting every Nth sample where N = original_sfreq / target_sfreq.

        Args:
            data (pd.DataFrame): Original accelerometer data
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

    def segment_data(self, data, epoch_length, sfreq):
        """
        Segment accelerometer data into fixed-length epochs.

        Divides continuous accelerometer data into non-overlapping windows of specified length.

        Args:
            data (pd.DataFrame): Continuous accelerometer data
            epoch_length (float): Length of each segment in seconds
            sfreq (float): Sampling frequency in Hz

        Returns:
            list: List of DataFrame segments, each of length epoch_length

        Note:
            - Only returns complete segments (partial segments at end are discarded)
            - Useful for activity classification and feature extraction

        Example:
            >>> segments = prepro.segment_data(data, epoch_length=5, sfreq=32)
            >>> print(f"Created {len(segments)} 5-second segments")
        """
        num_samples_per_epoch = int(epoch_length * sfreq)
        segments = [
            data.iloc[i:i + num_samples_per_epoch]
            for i in range(0, len(data), num_samples_per_epoch)
            if len(data.iloc[i:i + num_samples_per_epoch]) == num_samples_per_epoch
        ]
        return segments

    def preprocess_acc_data(self, epoch_length=5):
        """
        Complete accelerometer preprocessing pipeline.

        Applies the full preprocessing workflow:
        1. Noise removal (low-pass filtering)
        2. Missing value handling (interpolation)
        3. Downsampling to minimum sampling rate
        4. Normalization (z-score)
        5. Segmentation into epochs

        Args:
            epoch_length (float): Length of each epoch in seconds (default: 5)

        Returns:
            dict: Processed data for each stream containing:
                - 'data': Normalized continuous accelerometer data
                - 'epochs': List of segmented DataFrames
                - 'sfreq': Final sampling frequency

        Example:
            >>> prepro = PreProACC(acc_dataset)
            >>> processed = prepro.preprocess_acc_data(epoch_length=10)
            >>> for stream_id, stream_data in processed.items():
            ...     print(f"{stream_id}: {len(stream_data['epochs'])} epochs")

        Note:
            All streams are downsampled to the minimum sampling frequency across streams
        """
        processed_data = {}
        for stream_id, data_info in self.dataset.items():
            data, sfreq = data_info['data'], data_info['sfreq']

            # Noise removal
            data_no_noise = self.remove_noise(data, self.min_sfreq)

            # Handling missing values
            data_no_missing = self.handle_missing_values(data_no_noise)

            # Downsampling
            data_downsampled = self.downsample_data(data_no_missing, sfreq, self.min_sfreq)

            # Normalization
            normalized_data = self.normalize_data(data_downsampled)

            # Segmentation
            epochs = self.segment_data(normalized_data, epoch_length, self.min_sfreq)

            # Store processed data
            processed_data[stream_id] = {
                'data': normalized_data,
                'epochs': epochs,
                'sfreq': self.min_sfreq
            }

        return processed_data

    def plot_acc_data(self, dataset):
        """
        Generate 3-axis time-series plots of accelerometer data.

        Creates publication-quality plots showing all 3 axes (X, Y, Z) over time
        with proper labeling and sensor location information.

        Args:
            dataset (dict): Dataset to plot containing 'data' and metadata

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Creates one subplot per axis (X, Y, Z)
            - Infers sensor location from stream ID
            - Returns figures for saving or further customization

        Example:
            >>> figs, titles = prepro.plot_acc_data(processed_data)
            >>> for fig, title in zip(figs, titles):
            ...     fig.savefig(f"{title}.png", dpi=300)
        """
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']
            num_channels = data.shape[1]
            sensor_location = self.infer_sensor_location(stream)

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
            fig.suptitle(f'Accelerometer Data - {sensor_location.capitalize()} ({stream})')
            title = f'Accelerometer Data - {sensor_location.capitalize()} ({stream})'
            for i, channel in enumerate(['x', 'y', 'z']):
                axes[i].plot(data.index, data.iloc[:, i], label=f'ACC-{channel}')
                axes[i].set_title(f'Channel ACC-{channel}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(f'ACC-{i + 1} (μV)')
                axes[i].legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles

