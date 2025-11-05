"""
EEG Preprocessing Module

This module provides comprehensive preprocessing capabilities for EEG
(Electroencephalography) data including filtering, artifact removal, epoching,
and normalization.

Classes:
    PreProEEG: Main preprocessing class for EEG signals

Author: ProSense Contributors
Date: 2024
"""

import pandas as pd
import mne
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from load_data import LoadData


class PreProEEG:
    """
    Preprocessing class for EEG (Electroencephalography) data.

    This class provides a complete preprocessing pipeline for EEG signals including:
    - Downsampling and resampling
    - Filtering (bandpass, notch)
    - Artifact removal using ICA
    - Epoching
    - Baseline correction
    - Normalization and standardization

    Attributes:
        dataset (dict): Dictionary containing EEG datasets with stream IDs as keys
        min_sfreq (float): Minimum sampling frequency across all datasets

    Example:
        >>> eeg_data = {'stream_1': {'data': raw_eeg, 'sfreq': 256}}
        >>> prepro = PreProEEG(eeg_data)
        >>> prepro.apply_downsampling(max_sfreq=200)
        >>> prepro.apply_bandpass_filter(1.0, 40.0)
        >>> prepro.apply_artifact_removal()
    """

    def __init__(self, dataset):
        """
        Initialize the EEG preprocessing object.

        Args:
            dataset (dict): Dictionary of EEG datasets where each entry contains:
                - 'data': MNE Raw object with EEG data
                - 'sfreq': Sampling frequency in Hz

        Raises:
            ValueError: If dataset is empty or invalid
        """
        self.dataset = dataset
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

    def _check_and_filter(self, data, l_freq, h_freq):
        """
        Check frequency bounds and apply bandpass filter if valid.

        Args:
            data (mne.io.Raw): Raw EEG data
            l_freq (float): Low frequency bound (Hz)
            h_freq (float): High frequency bound (Hz)

        Returns:
            mne.io.Raw: Filtered data or original if filter cannot be applied

        Note:
            Skips filtering if frequencies violate Nyquist criterion
        """
        nyquist_freq = data.info['sfreq'] / 2
        if l_freq < 0 or h_freq >= nyquist_freq:
            print(f"Skipping bandpass filter for data with sfreq={data.info['sfreq']}")
            return data
        return data.filter(l_freq=l_freq, h_freq=h_freq)

    def apply_rereferencing(self, ref_channel='R_AUX'):
        """
        Apply common average referencing using a reference channel.

        Subtracts the reference channel signal from all EEG channels to reduce
        common mode noise and improve signal quality.

        Args:
            ref_channel (str): Name of the reference channel (default: 'R_AUX')

        Returns:
            dict: Updated dataset with re-referenced EEG data

        Note:
            Only processes streams that contain the specified reference channel
        """
        for stream_id, data_info in self.dataset.items():
            raw_data = data_info['data']
            if ref_channel in raw_data.ch_names:
                # Subtract the reference channel signal from all other EEG channels
                ref_data = raw_data.copy().pick_channels([ref_channel]).get_data()
                eeg_data = raw_data.copy().pick_types(eeg=True, exclude=[ref_channel]).get_data()
                re_referenced_data = eeg_data - ref_data

                # Update the dataset with the re-referenced data
                info = raw_data.copy().pick_types(eeg=True, exclude=[ref_channel]).info
                data_info['data'] = mne.io.RawArray(re_referenced_data, info)
        return self.dataset

    def apply_downsampling(self, max_sfreq=200):
        """
        Downsample EEG data to a maximum sampling frequency.

        Reduces the sampling rate of EEG data to save memory and computational
        resources while preserving signal information.

        Args:
            max_sfreq (int): Maximum desired sampling frequency in Hz (default: 200)

        Returns:
            dict: Updated dataset with downsampled data

        Raises:
            ValueError: If data becomes empty after resampling

        Note:
            Only downsamples if original frequency exceeds max_sfreq
        """
        for stream_id, data_info in self.dataset.items():
            raw_data = data_info['data']
            original_sfreq = raw_data.info['sfreq']
            print(f"Stream {stream_id}: Original Sampling Frequency = {original_sfreq} Hz")

            if original_sfreq > max_sfreq:
                # Resample the data
                raw_data_resampled = raw_data.copy().resample(max_sfreq, npad="auto")
                if raw_data_resampled._data.size == 0:
                    raise ValueError(f"Data in stream {stream_id} is empty after resampling.")
                data_info['data'] = raw_data_resampled
                data_info['sfreq'] = max_sfreq
                print(f"Stream {stream_id}: Data resampled to {max_sfreq} Hz")
            else:
                print(f"Stream {stream_id}: No resampling needed.")

        return self.dataset

    def apply_bandpass_filter(self, low_freq, high_freq):
        """
        Apply bandpass filter to retain specific frequency ranges.

        Filters EEG data to keep only frequencies within the specified band,
        removing both low-frequency drift and high-frequency noise.

        Args:
            low_freq (float): Lower cutoff frequency in Hz (e.g., 1.0 for delta)
            high_freq (float): Upper cutoff frequency in Hz (e.g., 40.0 for gamma)

        Returns:
            dict: Updated dataset with filtered data

        Example:
            >>> prepro.apply_bandpass_filter(1.0, 40.0)  # Keep 1-40 Hz
        """
        for data_info in self.dataset.values():
            data_info['data'] = self._check_and_filter(data_info['data'], low_freq, high_freq)
        return self.dataset

    def apply_notch_filter(self, freqs):
        """
        Apply notch filter to remove powerline noise and specific frequencies.

        Removes narrow-band noise at specified frequencies (commonly 50 or 60 Hz
        powerline interference).

        Args:
            freqs (float or list): Frequency or list of frequencies to remove (Hz)

        Returns:
            dict: Updated dataset with notch-filtered data

        Example:
            >>> prepro.apply_notch_filter(50)  # Remove 50 Hz powerline noise
            >>> prepro.apply_notch_filter([50, 100])  # Remove 50 Hz and harmonics
        """
        # Convert a single frequency to a list if necessary
        if isinstance(freqs, int) or isinstance(freqs, float):
            freqs = [freqs]

        for data_info in self.dataset.values():
            nyquist_freq = data_info['data'].info['sfreq'] / 2
            valid_freqs = [freq for freq in freqs if freq < nyquist_freq]
            if valid_freqs:
                data_info['data'].notch_filter(valid_freqs)
            else:
                print(f"Skipping notch filter for frequencies {freqs} due to low sampling rate.")
        return self.dataset

    def apply_artifact_removal(self):
        """
        Remove artifacts using Independent Component Analysis (ICA).

        Automatically identifies and removes artifact components from EEG data
        including eye blinks, muscle activity, and cardiac artifacts.

        Returns:
            dict: Updated dataset with artifacts removed

        Note:
            - Applies highpass filter (1 Hz) before ICA
            - Uses up to 15 ICA components (or number of channels if less)
            - Interpolates bad segments with NaN or Inf values
            - Random state fixed at 97 for reproducibility

        Warning:
            Automatic artifact removal may not be perfect. Consider manual review.
        """
        for data_info in self.dataset.values():
            data = data_info['data']
            data.filter(l_freq=1, h_freq=None)

            # Handle NaN or inf values in the data
            if np.any(np.isnan(data._data)) or np.isinf(np.sum(data._data)):
                # Option 1: Interpolate (if the amount of NaNs is not too high)
                data.interpolate_bads(reset_bads=True)

                # Option 2: Skip this segment if too many NaNs/Infs
                continue

            # Adjust the number of ICA components
            n_components = min(data.info['nchan'], 15)
            ica = ICA(n_components=n_components, random_state=97)
            ica.fit(data)
            ica.apply(data)
        return self.dataset

    def manual_component_selection(self):
        """
        Perform manual ICA component selection (requires user interaction).

        Allows users to manually select which ICA components to exclude from
        the data based on visual inspection.

        Returns:
            dict: Updated dataset with manually selected components removed

        Note:
            This method is currently not fully automated and requires manual
            component selection. User must uncomment and set ica.exclude.

        Warning:
            Not recommended for automated pipelines. Use apply_artifact_removal()
            for automatic processing.
        """
        # This method requires user interaction and is not fully automated
        for data_info in self.dataset.values():
            data = data_info['data']

            ica = ICA(n_components=15, random_state=97)
            ica.fit(data)

            # # User selects components to exclude
            # ica.exclude = user_selected_components

            ica.apply(data)
        return self.dataset

    def apply_epoching(self, epoch_duration):
        """
        Segment continuous EEG data into fixed-length epochs.

        Divides continuous EEG data into non-overlapping epochs of specified
        duration for time-locked analysis.

        Args:
            epoch_duration (float): Duration of each epoch in seconds

        Returns:
            dict: Updated dataset with 'epochs' added to each stream

        Example:
            >>> prepro.apply_epoching(epoch_duration=5.0)  # 5-second epochs

        Note:
            - Creates fixed-length events automatically
            - Rejects epochs marked by annotations
            - Epochs are stored in data_info['epochs']
        """
        for stream_id, data_info in self.dataset.items():
            data = data_info['data']
            events = mne.make_fixed_length_events(data, duration=epoch_duration)
            epochs = mne.Epochs(data, events, tmin=0, tmax=epoch_duration, baseline=None,
                                reject_by_annotation=True)
            data_info['epochs'] = epochs
        return self.dataset

    def apply_baseline_correction(self, baseline_channel='R_AUX'):
        """
        Apply baseline correction using a reference channel.

        Subtracts the mean of the baseline channel from all other channels
        to remove baseline drift and common mode interference.

        Args:
            baseline_channel (str): Name of the baseline/reference channel
                                   (default: 'R_AUX')

        Returns:
            dict: Updated dataset with baseline-corrected epochs

        Note:
            - Requires epochs to be created first (call apply_epoching())
            - Drops the baseline channel after correction
            - Only processes streams containing the baseline channel
        """
        for stream_id, data_info in self.dataset.items():
            epochs = data_info['epochs']
            if baseline_channel in epochs.ch_names:
                baseline_data = epochs.get_data(picks=baseline_channel)
                baseline_mean = np.nanmean(baseline_data, axis=-1, keepdims=True)
                for ch in epochs.ch_names:
                    if ch != baseline_channel:
                        epochs_data = epochs.get_data(picks=ch)
                        corrected_data = epochs_data - baseline_mean
                        epochs._data[epochs.ch_names.index(ch)] = corrected_data
            data_info['data'].drop_channels([baseline_channel])
            self.dataset[stream_id]['epochs'] = epochs
        return self.dataset

    def apply_rejection(self, threshold=100):
        """
        Reject epochs with amplitude exceeding threshold.

        Automatically removes epochs containing artifacts or extreme values
        that exceed the specified voltage threshold.

        Args:
            threshold (float): Maximum acceptable peak-to-peak amplitude in µV
                              (default: 100)

        Returns:
            dict: Updated dataset with bad epochs removed

        Note:
            Requires epochs to be created first (call apply_epoching())

        Example:
            >>> prepro.apply_rejection(threshold=150)  # Reject epochs > 150 µV
        """
        reject_criteria = dict(eeg=threshold)
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            epochs.drop_bad(reject=reject_criteria)
        return self.dataset

    def check_normality(self, alpha=0.05):
        """
        Test if EEG data follows a normal distribution.

        Performs statistical normality testing to determine if the data
        distribution is approximately Gaussian.

        Args:
            alpha (float): Significance level for normality test (default: 0.05)

        Returns:
            bool: True if data is normally distributed, False otherwise

        Note:
            - Uses scipy.stats.normaltest (D'Agostino-Pearson test)
            - Applies Bonferroni correction for multiple comparisons
            - Tests all streams in the dataset

        Example:
            >>> if prepro.check_normality():
            ...     prepro.apply_normalization()
            ... else:
            ...     prepro.apply_standardization()
        """
        p_values = []
        for data_info in self.dataset.values():
            data = data_info['data'].get_data()
            _, p = stats.normaltest(data.flatten())
            p_values.append(p)

        alpha_adjusted = alpha / len(p_values)
        is_normal = all(p > alpha_adjusted for p in p_values)
        return is_normal

    def apply_normalization(self):
        """
        Normalize epochs to [0, 1] range using min-max scaling.

        Scales each epoch's amplitude to the range [0, 1] by applying
        min-max normalization.

        Returns:
            dict: Updated dataset with normalized epochs

        Note:
            - Requires epochs to be created first
            - Formula: (x - min) / (max - min)
            - Use when data is not normally distributed
            - Alternative to standardization

        Example:
            >>> prepro.apply_normalization()  # Scale to [0, 1]
        """
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            data = epochs.get_data()
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            epochs = mne.EpochsArray(data_normalized, epochs.info)
            data_info['epochs'] = epochs
        return self.dataset

    def apply_standardization(self):
        """
        Standardize epochs to zero mean and unit variance (z-score).

        Applies z-score standardization to center data at zero with
        standard deviation of one.

        Returns:
            dict: Updated dataset with standardized epochs

        Note:
            - Requires epochs to be created first
            - Formula: (x - mean) / std
            - Use when data is normally distributed
            - Alternative to normalization

        Example:
            >>> prepro.apply_standardization()  # Z-score transform
        """
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            data = epochs.get_data()
            data_standardized = (data - np.mean(data)) / np.std(data)
            epochs = mne.EpochsArray(data_standardized, epochs.info)
            data_info['epochs'] = epochs
        return self.dataset

    def visualize_epochs(self):
        """
        Visualize epochs using interactive MNE plotting.

        Creates interactive plots showing the first 10 epochs for each stream
        in the dataset.

        Note:
            - Requires epochs to be created first
            - Opens interactive matplotlib windows
            - Displays 10 epochs per stream by default
        """
        for stream_id, data_info in self.dataset.items():
            epochs = data_info['epochs']
            epochs.plot(n_epochs=10, title=stream_id)

    def plot_eeg_data(self, dataset, title):
        """
        Generate multi-channel time-series plots of EEG data.

        Creates publication-quality plots showing all EEG channels over time
        with proper labeling and formatting.

        Args:
            dataset (dict): Dataset to plot (can be self.dataset or external)
            title (str): Title prefix for the plots

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Creates one subplot per channel
            - Automatically handles multiple streams
            - Returns figures for further customization or saving

        Example:
            >>> figs, titles = prepro.plot_eeg_data(prepro.dataset, "Filtered")
            >>> for fig, title in zip(figs, titles):
            ...     fig.savefig(f"{title}.png")
        """
        figs = []
        titles = []
        for stream, data_info in dataset.items():
            data = data_info['data']
            ch_names = data.info['ch_names']
            num_channels = len(ch_names)
            data_times = data.times
            data_array = data.get_data()

            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)
            fig.suptitle(f'{title} ({stream})')

            for i, channel in enumerate(ch_names):
                axes[i].plot(data_times, data_array[i, :], label=f'EEG-{channel}')
                axes[i].set_title(f'Channel: {channel}')
                axes[i].set_ylabel(f'EEG-{channel} (μV)')

            axes[-1].set_xlabel('Time (s)')
            fig.legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            figs.append(fig)
            titles.append(f'{title} ({stream})')
        return figs, titles