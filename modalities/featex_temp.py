import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class FeatExTEMP:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute_mean_temperature(self, data):
        return np.mean(data)

    def compute_max_temperature(self, data):
        return np.max(data)

    def compute_min_temperature(self, data):
        return np.min(data)

    def compute_temperature_variability(self, data):
        return np.std(data)

    def compute_rate_of_change(self, data):
        if len(data) > 1:
            derivative = np.diff(data)
            return np.mean(np.abs(derivative))
        else:
            return 0

    def extract_features(self):
        all_features = {}
        for stream, data_info in self.dataset.items():
            epochs = data_info['epochs']
            stream_features = []

            for epoch in epochs:
                features = {
                    "Min Temperature": self.compute_min_temperature(epoch),
                    "Mean Temperature": self.compute_mean_temperature(epoch),
                    "Max Temperature": self.compute_max_temperature(epoch),
                    "Temperature Variability": self.compute_temperature_variability(epoch),
                    "Rate of Change": self.compute_rate_of_change(epoch)
                }
                stream_features.append(features)

            all_features[stream] = stream_features

        return all_features

    def plot_features_over_epoch(self, all_features):
        figs = []
        titles = []
        for stream, epoch_features in all_features.items():
            num_features = len(epoch_features[0]) if epoch_features else 0
            fig, axes = plt.subplots(num_features, 1, figsize=(12, 6 * num_features), squeeze=False)
            fig.suptitle(f'Temperature Features Over Epochs - {stream}')
            title = f'Temperature Features Over Epochs - {stream}'

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
            fig.suptitle(f'Temperature Features Over Time - {stream}')
            title = f'Temperature Features Over Time - {stream}'

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



