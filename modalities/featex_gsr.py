import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


class FeatExGSR:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute_skin_conductance_level(self, data):
        return np.mean(data)

    def compute_skin_conductance_response(self, data):
        data_series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        peaks, _ = find_peaks(data_series, height=0.05)
        return len(peaks)

    def compute_amplitude_of_scrs(self, data):
        data_series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        peaks, properties = find_peaks(data_series, height=0.05)
        return np.mean(properties['peak_heights']) if peaks.size > 0 else 0

    def compute_variance_gsr(self, data):
        return np.var(data)

    def extract_features(self):
        all_features = {}
        for stream_id, data_info in self.dataset.items():
            epochs = data_info['epochs']
            stream_features = []

            for epoch in epochs:
                scl = self.compute_skin_conductance_level(epoch)
                scr_frequency = self.compute_skin_conductance_response(epoch)
                amplitude_scrs = self.compute_amplitude_of_scrs(epoch)
                gsr_var = self.compute_variance_gsr(epoch)

                epoch_features = {
                    "Skin Conductance Level (SCL)": scl,
                    "Skin Conductance Response (SCR) Frequency": scr_frequency,
                    "Amplitude of SCRs": amplitude_scrs,
                    "GSR Variance": gsr_var
                }
                stream_features.append(epoch_features)

            all_features[stream_id] = stream_features

        return all_features

    def plot_features_over_epoch(self, all_features):
        figs = []
        titles = []
        for stream, epoch_features in all_features.items():
            # Determine the number of features
            num_features = len(epoch_features[0]) if epoch_features else 0

            fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features), squeeze=False)
            fig.suptitle(f'GSR Features Over Epochs - {stream}')
            title = f'GSR Features Over Epochs - {stream}'

            if not num_features:
                print(f"No data to plot for stream {stream}")
                continue

            for i, feature_name in enumerate(epoch_features[0].keys()):
                feature_values = [epoch_feature[feature_name] for epoch_feature in epoch_features]

                ax = axes[i, 0]
                ax.plot(feature_values, label=feature_name)
                ax.set_title(f'{feature_name}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.legend(loc='upper left')

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles

    def plot_features_over_time(self, all_features, epoch_duration=5):
        figs = []
        titles = []
        for stream, epoch_features in all_features.items():
            start_time = pd.to_datetime(self.dataset[stream]['epochs'][0].index[0], unit='s')

            num_features = len(epoch_features[0]) if epoch_features else 0
            fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features), squeeze=False)
            fig.suptitle(f'GSR Features Over Time - {stream}')
            title = f'GSR Features Over Time - {stream}'

            if not num_features:
                print(f"No data to plot for stream {stream}")
                continue

            for i, feature_name in enumerate(epoch_features[0].keys()):
                times_vals = [
                    (start_time + pd.to_timedelta(idx * epoch_duration, unit='s'), epoch_feature[feature_name])
                    for idx, epoch_feature in enumerate(epoch_features)]

                times, vals = zip(*times_vals)

                ax = axes[i, 0]
                ax.plot(times, vals, label=feature_name)
                ax.set_title(f'{feature_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend(loc='upper left')

                for label in ax.get_xticklabels():
                    label.set_rotation(15)
                    label.set_ha('right')

            plt.tight_layout()
            figs.append(fig)
            titles.append(title)
        return figs, titles


