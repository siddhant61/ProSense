import pandas as pd
import mne
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from load_data import LoadData


class PreProEEG:
    def __init__(self, dataset):
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
        nyquist_freq = data.info['sfreq'] / 2
        if l_freq < 0 or h_freq >= nyquist_freq:
            print(f"Skipping bandpass filter for data with sfreq={data.info['sfreq']}")
            return data
        return data.filter(l_freq=l_freq, h_freq=h_freq)

    def apply_rereferencing(self, ref_channel='R_AUX'):
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
        for data_info in self.dataset.values():
            data_info['data'] = self._check_and_filter(data_info['data'], low_freq, high_freq)
        return self.dataset

    def apply_notch_filter(self, freqs):
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
        for stream_id, data_info in self.dataset.items():
            data = data_info['data']
            events = mne.make_fixed_length_events(data, duration=epoch_duration)
            epochs = mne.Epochs(data, events, tmin=0, tmax=epoch_duration, baseline=None,
                                reject_by_annotation=True)
            data_info['epochs'] = epochs
        return self.dataset

    def apply_baseline_correction(self, baseline_channel='R_AUX'):
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
        reject_criteria = dict(eeg=threshold)
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            epochs.drop_bad(reject=reject_criteria)
        return self.dataset

    def check_normality(self, alpha=0.05):
        p_values = []
        for data_info in self.dataset.values():
            data = data_info['data'].get_data()
            _, p = stats.normaltest(data.flatten())
            p_values.append(p)

        alpha_adjusted = alpha / len(p_values)
        is_normal = all(p > alpha_adjusted for p in p_values)
        return is_normal

    def apply_normalization(self):
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            data = epochs.get_data()
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            epochs = mne.EpochsArray(data_normalized, epochs.info)
            data_info['epochs'] = epochs
        return self.dataset

    def apply_standardization(self):
        for data_info in self.dataset.values():
            epochs = data_info['epochs']
            data = epochs.get_data()
            data_standardized = (data - np.mean(data)) / np.std(data)
            epochs = mne.EpochsArray(data_standardized, epochs.info)
            data_info['epochs'] = epochs
        return self.dataset

    def visualize_epochs(self):
        for stream_id, data_info in self.dataset.items():
            epochs = data_info['epochs']
            epochs.plot(n_epochs=10, title=stream_id)

    def plot_eeg_data(self, dataset, title):
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