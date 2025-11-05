"""
Gyroscope (GYRO) Preprocessing Module

This module provides preprocessing capabilities for gyroscope signals including
filtering, noise removal, normalization, downsampling, and segmentation.

Gyroscope signals measure angular velocity in three axes (X, Y, Z), commonly used for:
- Rotation and orientation tracking
- Gait analysis and balance assessment
- Activity recognition and gesture detection
- Motion pattern analysis
- Fall detection through sudden rotational changes
- Sports performance analysis

Classes:
    PreProGYRO: Main preprocessing class for 3-axis gyroscope signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt


class PreProGYRO:
    """
    Preprocessing class for Gyroscope (GYRO) data.

    This class provides a complete preprocessing pipeline for 3-axis gyroscope signals including:
    - Low-pass filtering (noise removal)
    - Downsampling
    - Normalization (z-score)
    - Segmentation into epochs
    - Sensor location inference

    Attributes:
        dataset (dict): Dictionary containing GYRO datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets

    Example:
        >>> gyro_data = {'stream_1': {'data': gyro_df, 'sfreq': 32}}
        >>> prepro = PreProGYRO(gyro_data)
        >>> processed = prepro.preprocess_gyro_data(epoch_length=5)
    """
    def __init__(self, dataset):
        """
        Initialize the GYRO preprocessing object.

        Args:
            dataset (dict): Dictionary of GYRO datasets where each entry contains:
                - 'data': pandas DataFrame with 3-axis gyroscope data (X, Y, Z)
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
        Apply Butterworth low-pass filter to gyroscope data.

        Uses scipy's filtfilt for zero-phase filtering to avoid signal distortion.
        Low-pass filters remove high-frequency noise while preserving rotational movements.

        Args:
            data (array-like): Gyroscope signal data to filter
            cutoff (float): Cutoff frequency in Hz
            sfreq (float): Sampling frequency in Hz
            order (int): Filter order (default: 5)

        Returns:
            array-like: Filtered signal

        Note:
            Typical cutoff for human motion is 5-20 Hz
            Angular velocity is measured in degrees/second or radians/second
        """
        nyq = 0.5 * sfreq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def remove_noise(self, data, sfreq):
        """
        Remove noise from gyroscope data using low-pass filtering.

        Performs comprehensive noise removal including:
        1. Converting data to numeric
        2. Handling inf/NaN values with forward/backward fill
        3. Applying low-pass filter to remove high-frequency noise

        Args:
            data (pd.DataFrame): Raw gyroscope data (3 axes: X, Y, Z)
            sfreq (float): Sampling frequency in Hz

        Returns:
            pd.DataFrame: Filtered gyroscope data

        Raises:
            ValueError: If data contains non-numeric columns after conversion
            ValueError: If data still contains NaN/inf after handling
            ValueError: If normalized cutoff frequency is invalid

        Note:
            Low-pass filter removes sensor noise and vibrations while preserving
            rotational movements (typically < 20 Hz)
        """
        # Convert all columns to numeric, turn non-convertible values into NaNs
        data_numeric = data.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

        # Check for completely non-numeric columns
        if data_numeric.isnull().all().any():
            raise ValueError("GYRO-Conversion to numeric resulted in non-numeric columns")

        # Replace inf values with NaN, then handle NaN values
        data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_ffill_bfill = data_numeric.ffill().bfill()

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
        """
        Normalize gyroscope data using z-score standardization.

        Transforms data to have zero mean and unit variance.

        Args:
            data (pd.DataFrame): Gyroscope data to normalize

        Returns:
            pd.DataFrame: Normalized data (mean=0, std=1)

        Note:
            Formula: (x - mean) / std
            Normalization makes data comparable across different sensors and locations
        """
        return (data - data.mean()) / data.std()

    def downsample_data(self, data, original_sfreq, target_sfreq):
        """
        Downsample gyroscope data to a lower sampling frequency.

        Reduces data rate by selecting every Nth sample where N = original_sfreq / target_sfreq.

        Args:
            data (pd.DataFrame): Original gyroscope data
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
            new_index = data.index[::factor]  # Assuming index is numeric and represents time
            downsampled_data.index = new_index
            return downsampled_data

    def segment_data(self, data, segment_length, sfreq):
        """
        Segment gyroscope data into fixed-length epochs.

        Divides continuous gyroscope data into non-overlapping windows of specified length.

        Args:
            data (pd.DataFrame): Continuous gyroscope data
            segment_length (float): Length of each segment in seconds
            sfreq (float): Sampling frequency in Hz

        Returns:
            list: List of DataFrame segments, each of length segment_length

        Note:
            - Only returns complete segments (partial segments at end are discarded)
            - Useful for activity classification and rotation analysis

        Example:
            >>> segments = prepro.segment_data(data, segment_length=5, sfreq=32)
            >>> print(f"Created {len(segments)} 5-second segments")
        """
        num_samples_per_segment = int(segment_length * sfreq)
        segments = [
            data.iloc[i:i + num_samples_per_segment]
            for i in range(0, len(data), num_samples_per_segment)
            if len(data.iloc[i:i + num_samples_per_segment]) == num_samples_per_segment
        ]
        return segments

    def infer_sensor_location(self, stream_id):
        """
        Infer the location of the gyroscope sensor from stream ID.

        Uses heuristics to determine sensor placement on the body.

        Args:
            stream_id (str): Stream identifier string

        Returns:
            str: Inferred location ('head', 'wrist', or 'unknown')

        Note:
            - 'muse' in stream_id → head-mounted sensor
            - Numeric characters (but not muse) → wrist-worn sensor
            - Otherwise → unknown location
        """
        if 'muse' in stream_id.lower():
            return 'head'
        elif any(char.isdigit() for char in stream_id) and not 'muse' in stream_id.lower():
            return 'wrist'
        else:
            return 'unknown'

    def preprocess_gyro_data(self, epoch_length=5):
        """
        Complete gyroscope preprocessing pipeline.

        Applies the full preprocessing workflow:
        1. Noise removal (low-pass filtering)
        2. Downsampling to minimum sampling rate
        3. Normalization (z-score)
        4. Segmentation into epochs

        Args:
            epoch_length (float): Length of each epoch in seconds (default: 5)

        Returns:
            dict: Processed data for each stream containing:
                - 'data': Normalized continuous gyroscope data
                - 'epochs': List of segmented DataFrames
                - 'sfreq': Final sampling frequency

        Example:
            >>> prepro = PreProGYRO(gyro_dataset)
            >>> processed = prepro.preprocess_gyro_data(epoch_length=10)
            >>> for stream_id, stream_data in processed.items():
            ...     print(f"{stream_id}: {len(stream_data['epochs'])} epochs")

        Note:
            All streams are downsampled to the minimum sampling frequency across streams
        """
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
        """
        Generate 3-axis time-series plots of gyroscope data.

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
            - Units are typically in degrees/second or radians/second

        Example:
            >>> figs, titles = prepro.plot_gyro_data(processed_data)
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
