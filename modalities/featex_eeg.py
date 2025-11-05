
"""
EEG Feature Extraction Module

This module provides comprehensive feature extraction capabilities for EEG signals including
frequency domain features, time-frequency analysis, coherence analysis, and statistical features.

EEG feature extraction transforms preprocessed epochs into meaningful metrics for:
- Cognitive state classification (attention, workload, drowsiness)
- Brain-computer interfaces (BCI)
- Sleep stage classification
- Seizure detection
- Mental workload assessment
- Emotion recognition

Key feature categories:
- Power band features (delta, theta, alpha, beta, gamma)
- Power band ratios (theta/delta, alpha/theta, etc.)
- Spectral entropy (signal complexity measure)
- Power spectral density (PSD) features
- Coherence features (connectivity between channels)
- Time-frequency features (TFR)
- Statistical features (mean, variance, skewness, kurtosis)

Classes:
    FeatExEEG: Main feature extraction class for EEG signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import mne
from mne import create_info
from mne.time_frequency import AverageTFR
from scipy import signal
from scipy.stats import kurtosis, skew


class FeatExEEG:
    """
    Feature extraction class for EEG data.

    Extracts a comprehensive set of frequency-domain, time-frequency, and statistical
    features from preprocessed EEG epochs for machine learning and analysis.

    Attributes:
        freq_bands (np.ndarray): Logarithmically spaced frequency bands for analysis

    Example:
        >>> featex = FeatExEEG()
        >>> features = featex.extract_features(preprocessed_dataset)
        >>> power_ratios = features['power_band_ratios']
    """
    def __init__(self):
        """Initialize EEG feature extractor with default frequency bands."""
        self.freq_bands = np.logspace(*np.log10([1, 35]), num=9)

    def extract_power_band_ratios(self, epochs):
        """
        Extract power band ratios from EEG epochs.

        Computes power in classical EEG frequency bands and their ratios,
        which are useful indicators of cognitive states.

        Args:
            epochs (mne.Epochs): Preprocessed EEG epochs

        Returns:
            dict: Nested dictionary with structure:
                {epoch_idx: {channel_name: {band_power, band_ratios}}}

        Bands extracted:
            - delta (1-4 Hz): Deep sleep, unconsciousness
            - theta (4-8 Hz): Drowsiness, meditation
            - alpha (8-12 Hz): Relaxed wakefulness
            - low_beta (12-15 Hz): Active thinking
            - high_beta (15-30 Hz): Alertness, anxiety

        Ratios computed:
            - theta/delta: Drowsiness indicator
            - alpha/theta: Relaxation vs drowsiness
            - beta/alpha: Alertness indicator

        Note:
            Higher theta/delta ratio indicates drowsiness
            Higher beta/alpha ratio indicates mental activation
        """
        power_band_ratios = {}

        # Define power bands (example bands)
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'low_beta': (12, 15),
            'high_beta': (15, 30)
        }

        # Iterate over epochs
        for i, epoch in enumerate(epochs):
            power_band_ratios[i] = {}
            for channel_idx, channel_name in enumerate(epochs.info['ch_names']):
                channel_data = epoch[channel_idx, :]  # Get data for current epoch and channel

                # Compute power spectral density (PSD) for each channel
                f, psd = signal.welch(channel_data, fs=epochs.info['sfreq'])

                # Compute power in each band for the current channel
                channel_power = {}
                for band_name, band_range in bands.items():
                    band_mask = (f >= band_range[0]) & (f < band_range[1])
                    band_power = np.trapz(psd[band_mask], x=f[band_mask])
                    channel_power[f'{band_name}_power'] = band_power

                # Compute power ratios for the current channel, checking for division by zero
                for ratio_name in ['theta_delta', 'alpha_delta', 'low_beta_delta', 'alpha_theta', 'low_beta_theta',
                                   'low_beta_alpha']:
                    numerator = channel_power.get(f'{ratio_name.split("_")[0]}_power', 0)
                    denominator = channel_power.get(f'{ratio_name.split("_")[1]}_power', 1)  # Avoid division by zero
                    channel_power[f'{ratio_name}_ratio'] = numerator / denominator if denominator != 0 else np.nan

                power_band_ratios[i][channel_name] = channel_power

        return power_band_ratios

    def extract_spectral_entropy(self, epochs, sfreq):
        """
        Extract spectral entropy from EEG epochs.

        Spectral entropy quantifies the complexity/irregularity of the EEG signal.
        Lower entropy indicates more regular/predictable signals (e.g., during sleep),
        higher entropy indicates more complex/irregular signals (e.g., during active cognition).

        Args:
            epochs (mne.Epochs): Preprocessed EEG epochs
            sfreq (float): Sampling frequency in Hz

        Returns:
            dict: {epoch_idx: {channel_name: spectral_entropy_value}}

        Note:
            - Range: 0 to log2(N) where N is number of frequency bins
            - Lower values: More regular signal (sleep, meditation)
            - Higher values: More complex signal (active thinking, eyes open)
            - Useful for drowsiness detection and cognitive load assessment
        """
        spectral_entropy = {}

        # Iterate over epochs
        for i, epoch in enumerate(epochs):
            spectral_entropy[i] = {}
            for channel_idx, channel_name in enumerate(epochs.info['ch_names']):
                channel_data = epoch[channel_idx, :]  # Get data for current epoch and channel

                # Compute power spectral density (PSD) for each channel
                f, psd = signal.welch(channel_data, fs=sfreq)

                # Compute spectral entropy for the current channel
                psd_norm = np.divide(psd, psd.sum())  # Normalize psd
                se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()  # Compute spectral entropy

                spectral_entropy[i][channel_name] = se

        return spectral_entropy

    def extract_psd_features(self, epochs, sfreq):
        psd_features = {}

        # Iterate over epochs
        for i, epoch in enumerate(epochs):
            psd_features[i] = {}
            for channel_idx, channel_name in enumerate(epochs.info['ch_names']):
                channel_data = epoch[channel_idx, :]  # Get data for current epoch and channel

                # Compute power spectral density (PSD) for each channel
                f, psd = signal.welch(channel_data, fs=sfreq)

                # Compute PSD features for the current channel
                psd_features[i][channel_name] = {
                    'max_power': np.max(psd),
                    'mean_power': np.mean(psd),
                    'median_power': np.median(psd)
                }

        return psd_features

    def extract_coherence_features(self, epochs, sfreq):
        coherence_features = {}

        # Iterate over epochs
        for i, epoch in enumerate(epochs):
            coherence_features[i] = {}
            for channel1_idx, channel1_name in enumerate(epochs.info['ch_names'][:-1]):
                for channel2_idx, channel2_name in enumerate(epochs.info['ch_names'][channel1_idx + 1:],
                                                             start=channel1_idx + 1):
                    channel1_data = epoch[channel1_idx, :]  # Get data for current epoch and channel1
                    channel2_data = epoch[channel2_idx, :]  # Get data for current epoch and channel2

                    # Compute coherence between the two channels
                    f, coh = signal.coherence(channel1_data, channel2_data, fs=sfreq)

                    # Compute mean coherence
                    mean_coh = np.mean(coh)

                    coherence_features[i][f'{channel1_name}_{channel2_name}'] = mean_coh

        return coherence_features

    def extract_time_frequency_features(self, epochs, freq_bands):
        # Get the sampling frequency
        sfreq = epochs.info['sfreq']

        # Nyquist frequency is half the sampling rate
        nyquist_freq = sfreq / 2.

        # Filter out frequencies higher than the Nyquist frequency
        freq_bands = [freq for freq in freq_bands if freq < nyquist_freq]

        if not freq_bands:
            raise ValueError("All frequency bands are above the Nyquist frequency. Adjust freq_bands or sampling rate.")

        # Compute the time-frequency representation using Morlet wavelets
        # Adjusting n_cycles for low-frequency data; typically 3-7 cycles are used for low frequencies
        n_cycles = [max(3, x / 2.) for x in freq_bands]
        tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freq_bands, n_cycles=n_cycles, return_itc=False)

        return tfr

    def extract_statistical_features(self, epochs):
        statistical_features = {}

        # Iterate over epochs
        for i, epoch in enumerate(epochs):
            statistical_features[i] = {}
            for channel_idx, channel_name in enumerate(epochs.info['ch_names']):
                channel_data = epoch[channel_idx, :]  # Get data for current epoch and channel

                # Compute statistical features for the current channel
                statistical_features[i][channel_name] = {
                    'mean': np.mean(channel_data),
                    'std': np.std(channel_data),
                    'variance': np.var(channel_data)
                }

        return statistical_features

    def epoch_mean(self, epochs):
        """Compute the mean of each epoch."""
        mean_values = np.mean(epochs, axis=2)
        return mean_values

    def epoch_variance(self, epochs):
        """Compute the variance of each epoch."""
        var_values = np.var(epochs, axis=2)
        return var_values

    def epoch_kurtosis(self, epochs):
        """Compute the kurtosis of each epoch."""
        kurtosis_values = kurtosis(epochs, axis=2)
        return kurtosis_values

    def epoch_skewness(self, epochs):
        """Compute the skewness of each epoch."""
        skewness_values = skew(epochs, axis=2)
        return skewness_values


    def extract_features(self, dataset):
        """
        Extract comprehensive feature set from EEG dataset.

        Main orchestration method that extracts all feature types from preprocessed
        EEG epochs. This is the primary entry point for EEG feature extraction.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {file_id: {'epochs': mne.Epochs, 'sfreq': float}}

        Returns:
            dict: Comprehensive feature dictionary with structure:
                {file_id: {
                    'power_band_ratios': {...},
                    'spectral_entropy': {...},
                    'psd_features': {...},
                    'coherence_features': {...},
                    'tfr_features': {...},
                    'statistical_features': {...}
                }}

        Features extracted:
            1. Power band ratios: Delta, theta, alpha, beta band powers and their ratios
            2. Spectral entropy: Signal complexity measure
            3. PSD features: Power spectral density characteristics
            4. Coherence features: Inter-channel connectivity measures
            5. Time-frequency features: Wavelet-based TFR analysis
            6. Statistical features: Mean, variance, skewness, kurtosis per epoch

        Example:
            >>> featex = FeatExEEG()
            >>> features = featex.extract_features(preprocessed_data)
            >>> # Access specific features
            >>> power_ratios = features['file1']['power_band_ratios']
            >>> entropy = features['file1']['spectral_entropy']

        Note:
            - Input must be preprocessed (filtered, epoched) EEG data
            - Returns nested dictionaries for flexible feature selection
            - Suitable for machine learning pipeline integration
        """
        all_features = {}


        for file, file_data in dataset.items():
            sfreq = file_data['sfreq']
            epochs = file_data['epochs']

            # Compute epoch-based features
            epoch_mean_values = self.epoch_mean(epochs)
            epoch_variance_values = self.epoch_variance(epochs)
            epoch_kurtosis_values = self.epoch_kurtosis(epochs)
            epoch_skewness_values = self.epoch_skewness(epochs)

            # Store the epoch-based features in a dictionary
            epoch_features = {
                "epoch_mean": epoch_mean_values,
                "epoch_variance": epoch_variance_values,
                "epoch_kurtosis": epoch_kurtosis_values,
                "epoch_skewness": epoch_skewness_values
            }

            power_band_ratios = self.extract_power_band_ratios(epochs)
            spectral_entropy_values = self.extract_spectral_entropy(epochs, sfreq)
            psd_features = self.extract_psd_features(epochs, sfreq)
            coherence_features = self.extract_coherence_features(epochs, sfreq)
            tfr_features = self.extract_time_frequency_features(epochs, freq_bands=self.freq_bands)
            statistical_features = self.extract_statistical_features(epochs)

            # Combine all features into a single dictionary for the file
            file_features = {
                "power_band_ratios": power_band_ratios,
                "spectral_entropy": spectral_entropy_values,
                "psd_features": psd_features,
                "coherence_features": coherence_features,
                "tfr_features": tfr_features,
                "statistical_features": statistical_features,
                "epoch_features": epoch_features,
                "total_epochs": len(epochs)
            }

            all_features[file] = file_features

        return all_features


    def plot_all_power_band_ratios(self, all_features, parameters):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            power_band_ratios = all_features[file]['power_band_ratios']

            # Assume each epoch is 5 seconds long
            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a time axis scaled to the total duration
            time = np.linspace(0, total_duration, len(power_band_ratios))

            # Get the list of ratios and channels
            channel_list = next(iter(power_band_ratios.values())).keys()

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_list), 1, figsize=(12, 8 * len(channel_list)), sharex=True)

            for ax, channel in zip(axs, channel_list):
                for idx, ratio_name in enumerate(parameters):
                    # Collect the ratio data for all epochs for the current channel
                    ratio_data = [epoch_data[channel][ratio_name] for epoch_data in power_band_ratios.values()]
                    color = colormap(idx % colormap.N)  # Assign a unique color based on the colormap
                    ax.plot(time, ratio_data, color=color, label=ratio_name)  # Set unique color for each plot

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{channel} Ratios')
                ax.legend()

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Power Band Ratios for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles

    def plot_specific_epochs_power_band_ratios(self, all_features, epoch_indices, band_name):
        figs = []
        titles = []

        for band in band_name:

            for file_name, features in all_features.items():
                power_band_ratios = features["power_band_ratios"]
                channel_names = list(next(iter(power_band_ratios.values())).keys())

                cmap = plt.get_cmap("tab10")

                for ch_idx, ch_name in enumerate(channel_names):
                    fig, ax = plt.subplots(figsize=(15, 5))
                    for epoch_idx in epoch_indices:
                        if epoch_idx in power_band_ratios.keys():
                            if ch_name in power_band_ratios[epoch_idx]:
                                if band in power_band_ratios[epoch_idx][ch_name].keys():
                                    ratio = power_band_ratios[epoch_idx][ch_name][f'{band}']
                                    ax.bar(epoch_idx, ratio, color=cmap(ch_idx / len(channel_names)))
                                else:
                                    print(f"No {band} in: {file_name}, Channel: {ch_name}, Epoch: {epoch_idx}")
                            else:
                                print(f"Channel {ch_name} not found in Epoch {epoch_idx} for file {file_name}")
                        else:
                            print(f"Epoch {epoch_idx} not found in file {file_name}")

                    ax.set_title(f'{file_name} - Channel {ch_name}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(f'{band} Power Ratio')
                    plt.tight_layout()
                    figs.append(fig)
                    titles.append(f"{file_name}_{ch_name}_Power_Band_Ratios")

        return figs, titles

    def plot_avg_power_band_ratios(self, all_features, parameters, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        avg_values = {ratio: np.zeros(max_epochs) for ratio in parameters}
        std_values = {ratio: np.zeros(max_epochs) for ratio in parameters}

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for _, features in all_features.items():
                feature_data = features['power_band_ratios']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    avg_values[ratio_name] += padded_ratio_values

            for ratio_name in parameters:
                avg_values[ratio_name] /= len(all_features)

            for _, features in all_features.items():
                feature_data = features['power_band_ratios']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    std_values[ratio_name] += (padded_ratio_values - avg_values[ratio_name]) ** 2

            for ratio_name in parameters:
                std_values[ratio_name] = np.sqrt(std_values[ratio_name] / len(all_features))

            for ratio_name in parameters:
                ax.plot(time_axis, avg_values[ratio_name], label=ratio_name,
                        color=colormap(parameters.index(ratio_name) % colormap.N))
                ax.fill_between(time_axis,
                                avg_values[ratio_name] - std_values[ratio_name],
                                avg_values[ratio_name] + std_values[ratio_name],
                                color=colormap(parameters.index(ratio_name) % colormap.N), alpha=0.3)

            ax.set_title(f"AF7 - Average Power band ratios with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.legend()
            figs.append(fig)
            titles.append(f"AF7_avg_with_std")

        return figs, titles


    def plot_all_spectral_entropies(self, all_features):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            spectral_entropies = all_features[file]['spectral_entropy']

            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a time axis scaled to the total duration
            time = np.linspace(0, total_duration, len(spectral_entropies))

            # Get the list of channels
            channel_list = next(iter(spectral_entropies.values())).keys()

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_list), 1, figsize=(12, 8 * len(channel_list)), sharex=True)

            for ax, channel in zip(axs, channel_list):
                # Collect the spectral entropy data for all epochs for the current channel
                entropy_data = [epoch_data[channel] for epoch_data in spectral_entropies.values()]
                ax.plot(time, entropy_data, color=colormap(0))  # Set unique color for each plot

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{channel} Spectral Entropy')

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Spectral Entropies for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles

    def plot_specific_epochs_spectral_entropy(self, all_features, epoch_indices):
        figs = []
        titles = []

        for file_name, features in all_features.items():
            power_band_ratios = features["spectral_entropy"]
            channel_names = list(next(iter(power_band_ratios.values())).keys())

            cmap = plt.get_cmap("tab10")

            for ch_idx, ch_name in enumerate(channel_names):
                fig, ax = plt.subplots(figsize=(15, 5))
                for epoch_idx in epoch_indices:
                    if epoch_idx in power_band_ratios.keys():
                        if ch_name in power_band_ratios[epoch_idx]:
                                entropy = power_band_ratios[epoch_idx][ch_name]

                                ax.bar(epoch_idx, entropy, color=cmap(ch_idx / len(channel_names)))
                        else:
                            print(f"Channel {ch_name} not found in Epoch {epoch_idx} for file {file_name}")
                    else:
                        print(f"Epoch {epoch_idx} not found in file {file_name}")

                ax.set_title(f'{file_name} - Channel {ch_name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'Spectral Entropy')
                plt.tight_layout()
                figs.append(fig)
                titles.append(f"{file_name}_{ch_name}_spectral_entropy")

        return figs, titles


    def plot_avg_spectral_entropy(self, all_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        # Initialize data structures to hold the average and standard deviation values
        avg_values = np.zeros(max_epochs)
        std_values = np.zeros(max_epochs)

        # Iterate over the channel names
        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Extract the spectral entropy values and compute the average
            all_entropy_values = []
            for _, features in all_features.items():
                spectral_entropy = features['spectral_entropy']
                entropy_values = [epoch_data[channel_name] for epoch_data in spectral_entropy.values()][:max_epochs]
                # Zero pad the entropy values to ensure they match the longest time axis
                padded_entropy_values = np.zeros(max_epochs)
                padded_entropy_values[:len(entropy_values)] = entropy_values
                all_entropy_values.append(padded_entropy_values)
                avg_values += padded_entropy_values

            avg_values /= len(all_features)

            # Compute the standard deviation
            for entropy_values in all_entropy_values:
                std_values += (entropy_values - avg_values) ** 2
            std_values = np.sqrt(std_values / len(all_features))

            # Plot the average and standard deviation values
            ax.plot(time_axis, avg_values, label='Average', color=colormap(0))
            ax.fill_between(time_axis, avg_values - std_values, avg_values + std_values, color=colormap(0), alpha=0.3)

            ax.set_title(f"{channel_name} - Average Spectral Entropy with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Spectral Entropy')
            ax.legend()
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles


    def plot_all_psd_values(self, all_features, parameters):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            psd_features = all_features[file]['psd_features']

            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a time axis scaled to the total duration
            time = np.linspace(0, total_duration, len(psd_features))

            # Get the list of ratios and channels
            channel_list = next(iter(psd_features.values())).keys()

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_list), 1, figsize=(12, 8 * len(channel_list)), sharex=True)

            for ax, channel in zip(axs, channel_list):
                for idx, ratio_name in enumerate(parameters):
                    # Collect the ratio data for all epochs for the current channel
                    ratio_data = [epoch_data[channel][ratio_name] for epoch_data in psd_features.values()]
                    color = colormap(idx % colormap.N)  # Assign a unique color based on the colormap
                    ax.plot(time, ratio_data, color=color, label=ratio_name)  # Set unique color for each plot

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{channel}')
                ax.legend()

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Power spectral density values for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles


    def plot_specific_epochs_psd_features(self, all_features, epoch_indices):
        c = 0
        figs = []
        titles = []
        for i in range(3, len(all_features.keys()), 3):
            file_names = list(all_features.keys())[c:i]  # Select the first 3 files
            if len(file_names) == 0:  # Skip this iteration if there are no files left
                continue
            c += 3
            # Define the channel names
            channel_names = ['AF7', 'AF8', 'TP9', 'TP10']

            # Define the feature names
            feature_names = ['max_power', 'mean_power', 'median_power']

            # Create a color map
            cmap = plt.get_cmap("tab10")

            # Iterate over the epoch indices
            for epoch_idx in epoch_indices:
                # Create a new figure for the current epoch
                fig, axs = plt.subplots(1, len(file_names), figsize=(15, 5), sharey=True)

                # Iterate over the file names
                for file_idx, file_name in enumerate(file_names):
                    # Get the PSD features for the current file
                    psd_features = all_features[file_name]['psd_features']

                    # Check if the current epoch index is valid for the current file
                    if epoch_idx not in psd_features:
                        continue

                    # Get the PSD features for the current epoch
                    epoch_data = psd_features[epoch_idx]

                    # Plot the PSD features for each channel
                    for ch_idx, ch_name in enumerate(channel_names):
                        # Get the PSD features for the current channel
                        ch_data = epoch_data[ch_name]

                        # Create a bar plot for the current channel
                        for feature_idx, feature_name in enumerate(feature_names):
                            axs[file_idx].bar(ch_idx + feature_idx / 4, ch_data[feature_name], width=0.25,
                                              color=cmap(feature_idx))

                    # Set the title of the subplot
                    axs[file_idx].set_title(f"{file_name} - Epoch {epoch_idx + 1}")

                    # Set the x-ticks and x-tick labels of the subplot
                    axs[file_idx].set_xticks(range(len(channel_names)))
                    axs[file_idx].set_xticklabels(channel_names, rotation=45)

                # Add a common y-axis label for all subplots
                fig.text(0.04, 0.5, 'PSD Features', va='center', rotation='vertical')

                # Create a custom legend
                custom_lines = [Line2D([0], [0], color=cmap(i), lw=4) for i in range(len(feature_names))]
                fig.legend(custom_lines, feature_names, loc='upper right')

                # Adjust the layout of the figure
                plt.tight_layout(rect=[0.04, 0, 0.86, 1])

                figs.append(fig)
                titles.append(f"file{c}-{i}_epoch{epoch_idx}")
        return figs, titles


    def plot_avg_psd_features(self, all_features, parameters, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        avg_values = {ratio: np.zeros(max_epochs) for ratio in parameters}
        std_values = {ratio: np.zeros(max_epochs) for ratio in parameters}

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for _, features in all_features.items():
                feature_data = features['psd_features']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    avg_values[ratio_name] += padded_ratio_values

            for ratio_name in parameters:
                avg_values[ratio_name] /= len(all_features)

            for _, features in all_features.items():
                feature_data = features['psd_features']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    std_values[ratio_name] += (padded_ratio_values - avg_values[ratio_name]) ** 2

            for ratio_name in parameters:
                std_values[ratio_name] = np.sqrt(std_values[ratio_name] / len(all_features))

            for ratio_name in parameters:
                ax.plot(time_axis, avg_values[ratio_name], label=ratio_name,
                        color=colormap(parameters.index(ratio_name) % colormap.N))
                ax.fill_between(time_axis,
                                avg_values[ratio_name] - std_values[ratio_name],
                                avg_values[ratio_name] + std_values[ratio_name],
                                color=colormap(parameters.index(ratio_name) % colormap.N), alpha=0.3)

            ax.set_title(f"{channel_name}- Average Psd features with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.legend()
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles


    def plot_all_coherence_values(self, all_features):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            coherence_features = all_features[file]['coherence_features']

            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a time axis scaled to the total duration
            time = np.linspace(0, total_duration, len(coherence_features))

            # Get the list of coherence pairs
            coherence_pairs = next(iter(coherence_features.values())).keys()

            # Create a subplot for each coherence pair
            fig, axs = plt.subplots(len(coherence_pairs), 1, figsize=(12, 8 * len(coherence_pairs)), sharex=True)

            for ax, coherence_pair in zip(axs, coherence_pairs):
                # Collect the coherence data for all epochs for the current pair of channels
                coherence_data = [epoch_data[coherence_pair] for epoch_data in coherence_features.values()]
                ax.plot(time, coherence_data, color=colormap(0))  # Set unique color for each plot

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{coherence_pair}')

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Coherence Values for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles


    def plot_specific_epochs_coherence_features(self, all_features, epoch_indices):
        c = 0
        figs = []
        titles = []
        for i in range(3, len(all_features.keys()), 3):
            file_names = list(all_features.keys())[c:i]  # Select the first 3 files
            if len(file_names) == 0:  # Skip this iteration if there are no files left
                continue
            # Create a color map
            cmap = plt.get_cmap("tab10")

            # Create a new figure for the current epoch
            fig, axs = plt.subplots(len(file_names), 1, figsize=(15, 5 * len(file_names)), sharey=True)

            # Iterate over the file names
            for file_idx, file_name in enumerate(file_names):
                # Get the coherence features for the current file
                coherence_features = all_features[file_name]['coherence_features']

                # Iterate over the epoch indices
                for epoch_idx in epoch_indices:
                    # Check if the current epoch index is valid for the current file
                    if epoch_idx not in coherence_features:
                        continue

                    # Get the coherence features for the current epoch
                    epoch_data = coherence_features[epoch_idx]

                    # Get the channel pairs
                    channel_pairs = list(epoch_data.keys())

                    # Get the coherence values
                    coherence_values = list(epoch_data.values())

                    # Create a bar plot for the current file with different colors for each bar
                    axs[file_idx].bar(
                        [x + (0.2 * (epoch_idx - min(epoch_indices))) - 0.4 + 0.1 for x in range(len(channel_pairs))],
                        coherence_values, width=0.15, color=cmap(epoch_idx - min(epoch_indices)))

                    # Set the title of the subplot
                    axs[file_idx].set_title(f"{file_name}")

                    # Rotate the x-tick labels
                    axs[file_idx].tick_params(axis='x', rotation=90)

                # Add a legend to the subplot
                legend_elements = [
                    Line2D([0], [0], color=cmap(epoch_idx - min(epoch_indices)), lw=4, label=f'Epoch {epoch_idx}') for
                    epoch_idx in epoch_indices]
                axs[file_idx].legend(handles=legend_elements)

            # Add a common y-axis label for all subplots
            fig.text(0.04, 0.5, 'Mean Coherence', va='center', rotation='vertical')

            # Adjust the layout of the figure
            plt.tight_layout(rect=[0.04, 0, 1, 1])

            figs.append(fig)
            titles.append(f"file{c}-{i}")
            c += 3
        return figs, titles


    def plot_avg_coherence_features(self, all_features, channel_pairs=[("AF7_AF8"), ("AF7_TP9"),
                                                                       ("AF7_TP10"), ("AF8_TP9"),
                                                                       ("AF8_TP10"), ("TP9_TP10")]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        # Initialize data structures to hold the average and standard deviation values
        avg_values = np.zeros(max_epochs)
        std_values = np.zeros(max_epochs)

        # Iterate over the channel pairs
        for channel_pair in channel_pairs:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Extract the coherence values and compute the average
            all_coherence_values = []
            for _, features in all_features.items():
                coherence_features = features['coherence_features']
                coherence_values = [epoch_data[channel_pair] for epoch_data in coherence_features.values()][:max_epochs]
                # Zero pad the coherence values to ensure they match the longest time axis
                padded_coherence_values = np.zeros(max_epochs)
                padded_coherence_values[:len(coherence_values)] = coherence_values
                all_coherence_values.append(padded_coherence_values)
                avg_values += padded_coherence_values

            avg_values /= len(all_features)

            # Compute the standard deviation
            for coherence_values in all_coherence_values:
                std_values += (coherence_values - avg_values) ** 2
            std_values = np.sqrt(std_values / len(all_features))

            # Plot the average and standard deviation values
            ax.plot(time_axis, avg_values, label='Average', color=colormap(0))
            ax.fill_between(time_axis, avg_values - std_values, avg_values + std_values, color=colormap(0),
                            alpha=0.3)

            channel_pair_str = f"{channel_pair}"
            ax.set_title(f"{channel_pair_str} - Average Coherence with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Coherence')
            ax.legend()
            figs.append(fig)
            titles.append(f"{channel_pair_str}_avg_with_std")

        return figs, titles

    def plot_avg_tfr_heatmap(self, all_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        """
        Plots the TFR values as a heatmap for all channels in a single figure using MNE's built-in plotting method.

        Parameters:
        - all_features: Dictionary containing features for multiple files. Each file's features should include an AverageTFR object.
        - channel_names: List of channel names to plot.

        Returns:
        - List of figures and corresponding titles.
        """
        figs = []
        titles = []

        for file, feature in all_features.items():
            tfr = feature['tfr_features']  # Extracting the AverageTFR object

            # Check if the extracted TFR is indeed an MNE AverageTFR object
            import mne
            if not isinstance(tfr, mne.time_frequency.AverageTFR):
                raise ValueError(f"The TFR for file {file} must be of type mne.time_frequency.AverageTFR.")

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_names), 1, figsize=(12, 8 * len(channel_names)), sharex=True,
                                    sharey=True)

            for ax, channel_name in zip(axs, channel_names):
                # Plot the heatmap for the specified channel
                tfr.plot([channel_name], baseline=(0., 0.5), mode='logratio', title=channel_name, axes=ax, show=False,
                         colorbar=False)

            # Add a title to the plot
            fig.suptitle(f"Time-Frequency Representation for {file}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title

            figs.append(fig)
            titles.append(f"{file}")

        return figs, titles


    def plot_all_tfr_values(self, all_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []
        for file, feature in all_features.items():
            tfr = all_features[file]['tfr_features']  # this is now a single AverageTFR object

            n_epochs = all_features[file]['total_epochs']

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_names), 1, figsize=(12, 8 * len(channel_names)), sharex=True)

            for ax, channel_name in zip(axs, channel_names):
                # Get the channel index
                channel_idx = channel_names.index(channel_name)

                # Compute the average TFR for the current channel across all frequencies
                avg_tfr = np.mean(tfr.data[channel_idx, :, :], axis=0)  # shape: (n_times,)

                # Create a new time array to represent the total time across all epochs
                total_times = np.arange(avg_tfr.size) * n_epochs * 5./ avg_tfr.size  # Assuming each epoch is 5 seconds

                # Plot the average TFR for the current channel
                ax.plot(total_times[40:-40], avg_tfr[40:-40], label=channel_name, color=colormap(channel_idx % colormap.N))

                ax.set_title(f"{channel_name} - Average TFR")
                ax.set_ylabel('Average TFR')
                ax.legend()

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Average Time-Frequency Representation for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles


    def plot_specific_epochs_tfr(self, all_features,
                                 channel_names=["AF7", "AF8", "TP9", "TP10"]):

        c = 0
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []
        for i in range(3, len(all_features.keys()), 3):
            file_names = list(all_features.keys())[c:i]  # Select the first 3 files
            c += 3

            # Create a new figure for each file
            fig, axs = plt.subplots(len(channel_names), len(file_names), figsize=(15, 10), sharey='row')

            # Iterate over the file names
            for file_idx, file_name in enumerate(file_names):
                # Get the tfr_features for the current file
                tfr = all_features[file_name]['tfr_features']

                # Get the indices of the freq_bands in the tfr object
                freq_indices = [np.where(np.isclose(tfr.freqs, freq, atol=1e-1))[0][0] for freq in self.freq_bands]

                # Iterate over the channel names
                for ch_idx, ch_name in enumerate(channel_names):
                    # Get the channel index
                    channel_idx = tfr.info['ch_names'].index(ch_name)

                    # Iterate over the frequency indices
                    for freq_idx in freq_indices:
                        # Get the TFR for the current channel and frequency
                        tfr_freq = tfr.data[channel_idx, freq_idx, :]

                        # Plot the TFR
                        axs[ch_idx, file_idx].plot(tfr.times[40:-40], tfr_freq[40:-40], label=f"Freq {tfr.freqs[freq_idx]:.2f} Hz",
                                                   color=colormap(freq_idx % colormap.N))

                    # Set the title of the subplot
                    axs[ch_idx, file_idx].set_title(f"{file_name} - {ch_name}")

                    # Set the labels of the subplot
                    axs[ch_idx, file_idx].set_xlabel('Time (s)')
                    axs[ch_idx, file_idx].set_ylabel('Power')
                    axs[ch_idx, file_idx].legend()

            # Adjust the layout of the figure
            plt.tight_layout()

            figs.append(fig)
            titles.append(f"files{c}-{i}")
            c += 3
        return figs, titles

    def plot_avg_tfr_values(self, all_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        import matplotlib.pyplot as plt
        from mne import create_info
        from mne.time_frequency import AverageTFR
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Fetching times and frequencies from one of the TFR objects in all_features
        tfr_times = all_features[list(all_features.keys())[0]]['tfr_features'].times
        freqs = all_features[list(all_features.keys())[0]]['tfr_features'].freqs

        # For each channel, we gather the TFR data from each file
        for channel_name in channel_names:
            channel_idx = all_features[list(all_features.keys())[0]]['tfr_features'].ch_names.index(channel_name)

            # Extract TFR data for the current channel from each file and store it in a list
            tfr_data_list = []
            for _, features in all_features.items():
                tfr_data = features['tfr_features'].data[channel_idx]
                # Ensure that the TFR data has the same shape (time x frequency)
                if tfr_data.shape == (len(tfr_times), len(freqs)):
                    tfr_data_list.append(tfr_data)

            # Skip channel if no valid TFR data is available
            if not tfr_data_list:
                continue

            # Average the TFR data across the files for each frequency and time point
            avg_tfr_data = np.mean(tfr_data_list, axis=0)

            # Create an info object for the specific channel
            channel_type = all_features[list(all_features.keys())[0]]['tfr_features'].info['chs'][channel_idx]['kind']
            ch_type_str = 'eeg' if channel_type == 2 else 'unknown'
            ch_info = create_info([channel_name], sfreq=200,
                                  ch_types=ch_type_str)

            # Create an AverageTFR object for the averaged data
            avg_tfr_obj = AverageTFR(info=ch_info,
                                     data=avg_tfr_data[None, ...],
                                     times=tfr_times,
                                     freqs=freqs,
                                     nave=len(tfr_data_list),
                                     comment=channel_name)

            # Plot the averaged heatmap
            fig_list = avg_tfr_obj.plot(baseline=(None, 0), mode='logratio',
                                        title=f"Averaged TFR of channel {channel_name}", show=False)

            # Directly append the figure to the figs list (to avoid nested lists)
            figs.extend(fig_list)
            titles.append(f"{channel_name}_averaged_TFR")

        return figs, titles

    def plot_all_stat_values(self, all_features, parameters):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            statistical_features = all_features[file]['statistical_features']

            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a time axis scaled to the total duration
            time = np.linspace(0, total_duration, len(statistical_features))

            # Get the list of ratios and channels
            channel_list = next(iter(statistical_features.values())).keys()

            # Create a subplot for each channel
            fig, axs = plt.subplots(len(channel_list), 1, figsize=(12, 8 * len(channel_list)), sharex=True)

            for ax, channel in zip(axs, channel_list):
                for idx, ratio_name in enumerate(parameters):
                    # Collect the ratio data for all epochs for the current channel
                    ratio_data = [epoch_data[channel][ratio_name] for epoch_data in statistical_features.values()]
                    color = colormap(idx % colormap.N)  # Assign a unique color based on the colormap
                    ax.plot(time, ratio_data, color=color, label=ratio_name)  # Set unique color for each plot

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{channel}')
                ax.legend()

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Stat values for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles


    def plot_specific_epochs_statistical_features(self,all_features, epoch_indices):
        c = 0
        figs = []
        titles = []
        for i in range(3, len(all_features.keys()), 3):
            file_names = list(all_features.keys())[c:i]  # Select the first 3 files
            if len(file_names) == 0:  # Skip this iteration if there are no files left
                continue
            c += 3
            # Create a color map
            cmap = plt.get_cmap("tab10")

            # Iterate over the epoch indices
            for epoch_idx in epoch_indices:
                # Create a new figure for the current epoch
                fig, axs = plt.subplots(1, len(file_names), figsize=(15, 5), sharey=True)

                # Iterate over the file names
                for file_idx, file_name in enumerate(file_names):
                    # Get the statistical features for the current file
                    statistical_features = all_features[file_name]['statistical_features']

                    # Check if the current epoch index is valid for the current file
                    if epoch_idx not in statistical_features:
                        continue

                    # Get the statistical features for the current epoch
                    epoch_data = statistical_features[epoch_idx]

                    # Get the channel names
                    channel_names = list(epoch_data.keys())

                    # Iterate over the statistical feature names
                    for stat_feature_name in ['mean', 'std', 'variance']:
                        # Get the statistical feature values
                        stat_feature_values = [ch_data[stat_feature_name] for ch_data in epoch_data.values()]

                        # Create a bar plot for the current file with different colors for each bar
                        axs[file_idx].bar(channel_names, stat_feature_values, color=cmap.colors[:len(channel_names)])

                        # Set the title of the subplot
                        axs[file_idx].set_title(f"{file_name} - Epoch {epoch_idx + 1} - {stat_feature_name}")

                        # Rotate the x-tick labels
                        axs[file_idx].tick_params(axis='x', rotation=90)

                # Add a common y-axis label for all subplots
                fig.text(0.04, 0.5, 'Statistical Features', va='center', rotation='vertical')

                # Adjust the layout of the figure
                plt.tight_layout(rect=[0.04, 0, 1, 1])

                figs.append(fig)
                titles.append(f"file{c}-{i}_epoch{epoch_idx}")
                c += 3
        return figs, titles


    def plot_avg_statistical_features(self, all_features, parameters, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])-20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        avg_values = {ratio: np.zeros(max_epochs) for ratio in parameters}
        std_values = {ratio: np.zeros(max_epochs) for ratio in parameters}

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for _, features in all_features.items():
                feature_data = features['statistical_features']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    avg_values[ratio_name] += padded_ratio_values

            for ratio_name in parameters:
                avg_values[ratio_name] /= len(all_features)

            for _, features in all_features.items():
                feature_data = features['statistical_features']

                for ratio_name in parameters:
                    ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_ratio_values = np.zeros(max_epochs)
                    padded_ratio_values[:len(ratio_values)] = ratio_values
                    std_values[ratio_name] += (padded_ratio_values - avg_values[ratio_name]) ** 2

            for ratio_name in parameters:
                std_values[ratio_name] = np.sqrt(std_values[ratio_name] / len(all_features))

            for ratio_name in parameters:
                ax.plot(time_axis, avg_values[ratio_name], label=ratio_name,
                        color=colormap(parameters.index(ratio_name) % colormap.N))
                ax.fill_between(time_axis,
                                avg_values[ratio_name] - std_values[ratio_name],
                                avg_values[ratio_name] + std_values[ratio_name],
                                color=colormap(parameters.index(ratio_name) % colormap.N), alpha=0.3)

            ax.set_title(f"AF7 - Average Statistical features with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.legend()
            figs.append(fig)
            titles.append(f"AF7_avg_with_std")

        return figs, titles


    def plot_all_epoch_stats(self, all_features, parameters, channel_names = ["AF7", "AF8", "TP9", "TP10"]):
        figs = []
        titles = []
        for file, feature in all_features.items():
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            epoch_features = all_features[file]['epoch_features']

            epoch_duration = 5
            n_epochs = feature['total_epochs']
            total_duration = n_epochs * epoch_duration

            # Create a subplot for each feature
            fig, axs = plt.subplots(len(parameters), 1, figsize=(12, 8 * len(parameters)), sharex=True)

            for ax, feature_name in zip(axs, parameters):
                # Collect the feature data for all epochs
                feature_data = epoch_features[feature_name]

                # Create a time axis scaled to the total duration
                time = np.linspace(0, total_duration, len(epoch_features[feature_name]))

                # Iterate over channels
                for channel_idx, channel_name in enumerate(channel_names):
                    ax.plot(time, feature_data[:, channel_idx], color=colormap(channel_idx % colormap.N),
                            label=channel_name)  # Set unique color for each channel

                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
                ax.set_ylabel(f'{feature_name}')
                ax.legend()

            axs[-1].set_xlabel('Time (s)')

            # Add a title to the plot
            fig.suptitle(f"Epoch Statistics for {file}", fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
            figs.append(fig)
            titles.append(f"{file}")
        return figs, titles


    def plot_specific_epochs_stats(self, all_features, epoch_indices, parameters):
        c = 0
        figs = []
        titles = []
        for i in range(3, len(all_features.keys()), 3):
            file_names = list(all_features.keys())[c:i]  # Select the first 3 files
            c += 3

            # Define the channel names
            channel_names = ['AF7', 'AF8', 'TP9', 'TP10']

            # Create a color map
            cmap = plt.get_cmap("tab10")

            # Iterate over the epoch indices
            for epoch_idx in epoch_indices:
                # Create a new figure for the current epoch
                fig, axs = plt.subplots(len(channel_names), len(file_names), figsize=(15, 10), sharey='row')

                # Iterate over the file names
                for file_idx, file_name in enumerate(file_names):
                    # Get the epoch_features for the current file
                    epoch_features = all_features[file_name]['epoch_features']

                    # Iterate over the channel names
                    for ch_idx, ch_name in enumerate(channel_names):
                        # Get the channel index
                        channel_idx = channel_names.index(ch_name)

                        # Iterate over the parameters
                        for param_idx, parameter in enumerate(parameters):
                            # Get the data for the current parameter, epoch, and channel
                            data = epoch_features[parameter][epoch_idx, channel_idx]

                            # Plot the data
                            axs[ch_idx, file_idx].bar(param_idx, data, color=cmap(param_idx))

                        # Set the title of the subplot
                        axs[ch_idx, file_idx].set_title(f"{file_name} - {ch_name}")

                        # Set the x-ticks and x-tick labels of the subplot
                        axs[ch_idx, file_idx].set_xticks(range(len(parameters)))
                        axs[ch_idx, file_idx].set_xticklabels(parameters, rotation=45)

                # Adjust the layout of the figure
                plt.tight_layout()

                figs.append(fig)
                titles.append(f"file{c}{i}_epoch{epoch_idx}")
                c += 3
        return figs, titles


    def plot_avg_epoch_features(self, all_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        epoch_features_names = ['epoch_mean', 'epoch_kurtosis', 'epoch_skewness']
        figs = []
        titles = []

        # Determine the longest time axis
        max_epochs = max([features['total_epochs'] for _, features in all_features.items()])-20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        # Iterate over the channel names
        for channel_name in channel_names:
            fig, axs = plt.subplots(len(epoch_features_names), 1, figsize=(12, 8 * len(epoch_features_names)),
                                    sharex=True)

            # Iterate over the epoch features
            for ax, epoch_feature_name in zip(axs, epoch_features_names):
                # Initialize data structures to hold the average and standard deviation values
                avg_values = np.zeros(max_epochs)
                std_values = np.zeros(max_epochs)
                all_feature_values = []

                # Extract the epoch feature values and compute the average
                for _, features in all_features.items():
                    epoch_features = features['epoch_features']
                    feature_values = epoch_features[epoch_feature_name][:, channel_names.index(channel_name)][:max_epochs]
                    padded_feature_values = np.zeros(max_epochs)
                    padded_feature_values[:len(feature_values)] = feature_values
                    all_feature_values.append(padded_feature_values)
                    avg_values += padded_feature_values

                avg_values /= len(all_features)

                # Compute the standard deviation
                for feature_values in all_feature_values:
                    std_values += (feature_values - avg_values) ** 2
                std_values = np.sqrt(std_values / len(all_features))

                # Plot the average and standard deviation values
                ax.plot(time_axis, avg_values, label='Average', color=colormap(0))
                ax.fill_between(time_axis, avg_values - std_values, avg_values + std_values, color=colormap(0),
                                alpha=0.3)

                ax.set_title(f"{channel_name} - {epoch_feature_name} with Std Deviation")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(epoch_feature_name.split('_')[1].capitalize())
                ax.legend()

            plt.tight_layout()
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles

    def plot_features(self, all_features, channels=['TP9', 'AF7', 'AF8', 'TP10']):
        for file, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                if isinstance(epoch_features, dict):  # To handle dictionaries
                    num_epochs = len(epoch_features)
                    for epoch in epoch_features.values():
                        for feature, values in epoch.items():
                            if isinstance(values, dict):  # Checking if the values are dictionary type
                                self._plot_feature_graph(num_epochs, feature, values, channels, file, feature_type)

    def _plot_feature_graph(self, num_epochs, feature, values, channels, file, feature_type):
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, len(channels) * 4))
        fig.suptitle(f'{file} - {feature.capitalize()} over Epochs - {feature_type}', fontsize=16)

        for j, channel in enumerate(channels):
            channel_feature_values = [values[channel][feature] for _ in range(num_epochs)]

            ax = axes[j]
            ax.plot(range(num_epochs), channel_feature_values,
                    label=f'{channel.upper()} Channel - {feature.capitalize()}')
            ax.set_title(f'{channel.upper()} Channel')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{feature.capitalize()} Value')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_features_over_time(self, all_features, epoch_duration=5, channels=['TP9', 'AF7', 'AF8', 'TP10']):
        for file, feature_data in all_features.items():
            for feature_type, epoch_features in feature_data.items():
                if isinstance(epoch_features, dict):  # To handle dictionaries
                    num_epochs = len(epoch_features)
                    for epoch in epoch_features.values():
                        for feature, values in epoch.items():
                            if isinstance(values, dict):  # Checking if the values are dictionary type
                                self._plot_feature_over_time(num_epochs, epoch_duration, feature, values, channels,
                                                             file, feature_type)

    def _plot_feature_over_time(self, num_epochs, epoch_duration, feature, values, channels, file, feature_type):
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, len(channels) * 4))
        fig.suptitle(f'{file} - {feature.capitalize()} over Time - {feature_type}', fontsize=16)

        for j, channel in enumerate(channels):
            times = [i * epoch_duration for i in range(num_epochs)]
            channel_feature_values = [values[channel][feature] for _ in range(num_epochs)]

            ax = axes[j]
            ax.plot(times, channel_feature_values, label=f'{channel.upper()} Channel - {feature.capitalize()}')
            ax.set_title(f'{channel.upper()} Channel')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{feature.capitalize()} Value')
            ax.legend()

        plt.tight_layout()
        plt.show()