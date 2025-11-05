"""
Galvanic Skin Response (GSR/EDA) Feature Extraction Module

This module provides feature extraction capabilities for GSR/EDA signals including
tonic and phasic component analysis for sympathetic nervous system activity monitoring.

GSR (also called EDA - Electrodermal Activity) measures skin conductance changes
caused by sweat gland activity controlled by the sympathetic nervous system.

GSR features are useful for:
- Stress and anxiety detection (increased SCL and SCR frequency)
- Emotional arousal measurement (amplitude of SCRs)
- Cognitive load assessment (elevated baseline SCL)
- Lie detection and deception analysis
- Pain assessment
- Sleep stage classification
- PTSD and phobia research
- Human-computer interaction (user engagement)

Signal components:
- SCL (Skin Conductance Level): Tonic/baseline component reflecting general arousal
- SCR (Skin Conductance Response): Phasic component showing rapid responses to stimuli

Classes:
    FeatExGSR: Main feature extraction class for GSR/EDA signals

Author: ProSense Contributors
Date: 2024
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


class FeatExGSR:
    """
    Feature extraction class for Galvanic Skin Response (GSR/EDA) data.

    Extracts both tonic (SCL) and phasic (SCR) features from preprocessed
    GSR epochs for sympathetic nervous system activity analysis.

    Attributes:
        dataset (dict): Preprocessed GSR dataset with epochs

    Example:
        >>> featex = FeatExGSR(preprocessed_gsr_data)
        >>> features = featex.extract_features()
        >>> print(features['stream_1'][0])  # First epoch features
    """

    def __init__(self, dataset):
        """
        Initialize GSR feature extractor.

        Args:
            dataset (dict): Preprocessed dataset with structure:
                {stream_id: {'epochs': [epoch_dataframes], 'sfreq': float}}
        """
        self.dataset = dataset

    def compute_skin_conductance_level(self, data):
        """
        Compute Skin Conductance Level (SCL) - tonic component.

        SCL represents the baseline/tonic level of electrodermal activity,
        reflecting general arousal state and sympathetic nervous system activation.

        Args:
            data (array-like or DataFrame): GSR data for one epoch

        Returns:
            float: Mean skin conductance level (microsiemens, µS)

        Note:
            - Higher SCL indicates higher general arousal/stress
            - Reflects slower-changing baseline activity
            - Useful for detecting sustained stress or cognitive load
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.mean(data))

    def compute_skin_conductance_response(self, data):
        """
        Compute Skin Conductance Response (SCR) frequency - phasic component.

        SCR represents rapid, event-related electrodermal responses to stimuli,
        reflecting phasic sympathetic nervous system activation.

        Args:
            data (array-like or DataFrame): GSR data for one epoch

        Returns:
            int: Number of SCR peaks detected (count of phasic responses)

        Note:
            - Peaks detected with minimum height threshold of 0.05 µS
            - Higher SCR frequency indicates more stimulus-driven arousal
            - Reflects rapid responses to emotional or cognitive events
            - Typical SCR has 1-5 second rise time and 5-10 second recovery
        """
        data_series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        peaks, _ = find_peaks(data_series, height=0.05)
        return len(peaks)

    def compute_amplitude_of_scrs(self, data):
        """
        Compute average amplitude of Skin Conductance Responses (SCRs).

        Measures the average magnitude of phasic responses, indicating
        intensity of sympathetic nervous system activation.

        Args:
            data (array-like or DataFrame): GSR data for one epoch

        Returns:
            float: Mean amplitude of detected SCR peaks (µS), 0 if no peaks

        Note:
            - Higher amplitudes indicate stronger emotional/cognitive responses
            - Typical SCR amplitude ranges from 0.05 to 1.0 µS
            - Amplitude correlates with arousal intensity
            - Returns 0 when no peaks are detected
        """
        data_series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
        peaks, properties = find_peaks(data_series, height=0.05)
        return float(np.mean(properties['peak_heights'])) if peaks.size > 0 else 0.0

    def compute_variance_gsr(self, data):
        """
        Compute variance of GSR signal.

        Measures overall variability in electrodermal activity, capturing
        both tonic and phasic fluctuations.

        Args:
            data (array-like or DataFrame): GSR data for one epoch

        Returns:
            float: Variance of GSR signal (µS²)

        Note:
            - Higher variance indicates more dynamic electrodermal activity
            - Reflects both SCL changes and SCR frequency/amplitude
            - Useful for detecting periods of high arousal or stress
        """
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        return float(np.var(data))

    def extract_features(self):
        """
        Extract comprehensive GSR features from all epochs.

        Main entry point for GSR/EDA feature extraction. Processes all streams
        and epochs to extract tonic (SCL) and phasic (SCR) features.

        Returns:
            dict: Features for each stream with structure:
                {stream_id: [
                    {
                        'Skin Conductance Level (SCL)': float,
                        'Skin Conductance Response (SCR) Frequency': int,
                        'Amplitude of SCRs': float,
                        'GSR Variance': float
                    },
                    ...  # One dict per epoch
                ]}

        Example:
            >>> featex = FeatExGSR(dataset)
            >>> features = featex.extract_features()
            >>> # Access features for first epoch of stream_1
            >>> epoch_0_features = features['stream_1'][0]
            >>> print(f"SCL: {epoch_0_features['Skin Conductance Level (SCL)']}")
            >>> print(f"SCR freq: {epoch_0_features['Skin Conductance Response (SCR) Frequency']}")

        Note:
            Each epoch yields 4 features combining tonic and phasic components
            for comprehensive sympathetic nervous system activity assessment.
        """
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
        """
        Plot GSR features across epochs.

        Creates time-series visualization of extracted GSR features showing
        how SCL, SCR frequency, amplitude, and variance change across epochs.

        Args:
            all_features (dict): Extracted features from extract_features()

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - Each feature plotted in separate subplot
            - Useful for visualizing arousal patterns over time
            - SCL trends show sustained arousal changes
            - SCR patterns show event-related responses
        """
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
        """
        Plot GSR features over absolute time.

        Creates time-series visualization with real timestamps showing temporal
        patterns in sympathetic nervous system activity.

        Args:
            all_features (dict): Extracted features from extract_features()
            epoch_duration (float): Length of each epoch in seconds (default: 5)

        Returns:
            tuple: (figs, titles) where:
                - figs: List of matplotlib Figure objects
                - titles: List of corresponding plot titles

        Note:
            - X-axis shows absolute timestamps
            - Useful for correlating arousal with external events
            - Can reveal circadian patterns in baseline SCL
            - SCR bursts visible during stress/emotional periods
        """
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


