import json
import os
from datetime import timedelta
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt



def save_figures(figs, titles, name, save_path):
    save_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, (fig, title) in enumerate(zip(figs, titles)):
        # Replace spaces and special characters in the title with underscores
        title = title.replace(' ', '_')
        for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            title = title.replace(char, '_')
        # Generate the file name using the provided name, index, and title
        file_name = f'{name}_{i}_{title}.png'
        file_path = os.path.join(save_path, file_name)
        fig.savefig(file_path)
        plt.close(fig)


class PreProData:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def load_datasets(self, logs_directory):
        datasets = {}
        markers = {}
        events = {}
        event_logs, event_markers = self.load_event_logs_and_markers(logs_directory)

        # Creating a mapping from sensor ID to log key
        log_mapping = {}
        for key in event_logs.keys():
            muse_sensor_id = key.split('_')[1].replace('m', '')
            empatica_sensor_id = key.split('_')[2].replace('e', '')
            log_mapping[muse_sensor_id] = key
            log_mapping[empatica_sensor_id] = key

            print(f"Log key: {key}, Muse ID: {muse_sensor_id}, Empatica ID: {empatica_sensor_id}")

        for file in os.listdir(self.dataset_folder):
            if file.endswith('.csv') and not file.endswith('_events.csv'):
                stream_id = os.path.splitext(file)[0]
                sensor_id = stream_id.split('_')[0].lower()
                data_path = os.path.join(self.dataset_folder, file)

                if 'muses-' in sensor_id:
                    sensor_id = sensor_id.replace('muses-', '')

                # Check if the current sensor ID is in the log_mapping
                if sensor_id in log_mapping.keys():
                    log_key = log_mapping[sensor_id]

                    # Proceed with synchronization and trimming
                    if 'ACC' in file and 'Muse' not in file:
                        print(f"Processing {stream_id} for {log_key}")
                        df = pd.read_csv(data_path, index_col=0)
                        markers[log_key], spikes = self.synchronize_markers_with_acc(df, event_markers[log_key], 0)
                        figs, titles = self.visual_inspection(stream_id, df, markers[log_key])
                        save_figures(figs, titles, 'sync_events', f"{self.dataset_folder}/Plots/SYNC")
                        figs, titles = self.plot_acc_data_with_markers(stream_id, df, markers[log_key], spikes)
                        save_figures(figs, titles, 'aligned_data', f"{self.dataset_folder}/Plots/SYNC")

        for file in os.listdir(self.dataset_folder):
            if file.endswith('.csv') and not file.endswith('_events.csv'):
                stream_id = os.path.splitext(file)[0]
                sensor_id = stream_id.split('_')[0].lower()
                data_path = os.path.join(self.dataset_folder, file)
                metadata_path = os.path.splitext(data_path)[0] + '_metadata.json'


                if 'muses-' in sensor_id:
                    sensor_id = sensor_id.replace('muses-', '')

                log_key = log_mapping[sensor_id]
                # print(f"Processing {stream_id} for {log_key}")

                events[stream_id] = markers[log_key]

                df = pd.read_csv(data_path, index_col=0)

                self.start_time = df.index[0]

                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                try:
                    with open(metadata_path, 'r') as meta_file:
                        metadata = json.load(meta_file)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file {metadata_path}: {e}")

                start_time, end_time = markers[log_key][0][1], markers[log_key][-1][1]

                print(f"Start time: {start_time}, End time: {end_time}")


                if 'EEG' not in file:
                    df = self.trim_data_to_events(df, start_time, end_time)
                    datasets[stream_id] = {
                        'data': df,
                        'sfreq': metadata['sfreq'],
                        'markers': markers[log_key]
                    }

                else:
                    # Convert DataFrame to MNE Raw object
                    data = self._load_mne_raw(df, metadata['sfreq'])

                    # Add event markers to the MNE Raw object
                    data = self.add_event_markers(data, markers[log_key], start_time)

                    data = self.trim_raw_data(data, markers[log_key])

                    datasets[stream_id] = {
                        'data': data,
                        'sfreq': metadata['sfreq'],
                        'markers': markers[log_key]
                    }


        return events, datasets, log_mapping

    def trim_raw_data(self, raw_data, markers):
        if not markers:
            return raw_data  # Return original data if no markers are provided

        # Convert timestamps to seconds since the start of the recording
        meas_start = pd.to_datetime(self.start_time) if isinstance(self.start_time, str) else self.start_time
        first_marker_time = markers[0][1]
        last_marker_time = markers[-1][1] + markers[-1][3]

        first_marker_secs = (first_marker_time - meas_start).total_seconds()
        last_marker_secs = (last_marker_time - meas_start).total_seconds()

        # Check if first_marker_secs or last_marker_secs is NaN
        if pd.isna(first_marker_secs) or pd.isna(last_marker_secs):
            print("Error: NaN detected in marker seconds.")
            return raw_data

        # Ensure that tmax is within the range of the raw data
        max_time_in_raw_data = raw_data.times[-1]
        last_marker_secs = min(last_marker_secs, max_time_in_raw_data)

        print(f"Cropping data from {first_marker_secs} to {last_marker_secs} seconds.")

        # Crop the raw data between the first and last markers
        try:
            trimmed_data = raw_data.copy().crop(tmin=first_marker_secs, tmax=last_marker_secs)
            return trimmed_data
        except Exception as e:
            print(f"Error during cropping: {e}")
            return raw_data

    def visual_inspection(self, stream_id, acc_data, event_markers):
        figs = []
        titles = []
        # Convert acc_data index to pd.Timestamp if it's in string format
        if isinstance(acc_data.index, pd.Index) and acc_data.index.dtype == 'object':
            acc_data.index = pd.to_datetime(acc_data.index)

        sync_markers = [m for m in event_markers if 'SYNC' in m[0] and 'instr' not in m[0]]
        if len(sync_markers) >= 2:
            first_sync_marker, second_sync_marker = sync_markers[0], sync_markers[1]

            # Extracting timestamps and durations for the sync events
            first_sync_time = pd.to_datetime(first_sync_marker[1])
            first_duration = first_sync_marker[3].total_seconds()
            second_sync_time = pd.to_datetime(second_sync_marker[1])
            second_duration = second_sync_marker[3].total_seconds()

            # Setting the window for visualization
            window_start = first_sync_time
            window_end = second_sync_time + pd.Timedelta(seconds=second_duration)

            # Filtering the accelerometer data for the selected window
            window_data = acc_data[(acc_data.index >= window_start) & (acc_data.index <= window_end)]

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(window_data.index, window_data, label=['X', 'Y', 'Z'])
            ax.axvline(x=first_sync_time, color='red', linestyle='--', label='First SYNC Event')
            ax.axvspan(first_sync_time, first_sync_time + pd.Timedelta(seconds=first_duration), color='red', alpha=0.3)
            ax.axvline(x=second_sync_time, color='green', linestyle='--', label='Second SYNC Event')
            ax.axvspan(second_sync_time, second_sync_time + pd.Timedelta(seconds=second_duration), color='green',
                       alpha=0.3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude (µV)')
            title = f'{stream_id} Data around SYNC Events'
            ax.set_title(title)
            ax.legend(loc='upper right')

            figs.append(fig)
            titles.append(title)

            return figs, titles
        else:
            print("Not enough SYNC markers found.")


    def plot_acc_data_with_markers(self, stream_id, acc_data, sync_markers, detected_spikes):
        """
        Plot accelerometer data with SYNC event markers and detected spikes.

        :param acc_data: DataFrame containing accelerometer data with a DateTime index.
        :param sync_markers: List of SYNC event markers after alignment.
        :param detected_spikes: Dictionary of lists of timestamps where spikes were detected in the accelerometer data.
        """
        figs = []
        titles = []

        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot the accelerometer data
        for column in acc_data.columns:
            plt.plot(acc_data.index, acc_data[column], label=column)

        # Mark the SYNC event markers
        for marker, marker_time, category, duration in sync_markers:
            if 'SYNC' in marker and 'instr' not in marker:
                plt.axvline(x=marker_time, color='blue', linestyle='-', label='Aligned SYNC Marker')

        # Mark the detected spikes in accelerometer data
        for spike_key in detected_spikes:  # Iterate over each set of spikes (e.g., 'first_spikes', 'second_spikes')
            for spike_time in detected_spikes[spike_key]:
                plt.axvline(x=spike_time, color='green', linestyle='--', label=f'Detected Spike ({spike_key})')

        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude (µV)')
        title = f'{stream_id} Data with SYNC Markers and Detected Spikes'
        ax.set_title(title)
        ax.legend(loc='upper right')

        figs.append(fig)
        titles.append(title)

        return figs, titles

    def synchronize_markers_with_acc(self, acc_data, event_markers, start_offset_seconds):
        if not isinstance(acc_data.index, pd.DatetimeIndex):
            acc_data.index = pd.to_datetime(acc_data.index)

        recording_start_time = acc_data.index[0]
        start_offset = recording_start_time + pd.Timedelta(seconds=start_offset_seconds)
        acc_data = acc_data[acc_data.index > start_offset]

        sync_markers = [m for m in event_markers if 'SYNC' in m[0] and 'instr' not in m[0]]
        if len(sync_markers) < 2:
            return event_markers  # Insufficient SYNC markers to synchronize

        first_sync_marker, second_sync_marker = sync_markers[:2]
        first_spikes = self.detect_spikes_in_window(acc_data, pd.to_datetime(first_sync_marker[1]), window=first_sync_marker[3].total_seconds())
        second_spikes = self.detect_spikes_in_window(acc_data, pd.to_datetime(second_sync_marker[1]), window=second_sync_marker[3].total_seconds())

        first_spikes, second_spikes =  self.detect_spikes_for_sync_events(acc_data, sync_markers[:2])

        # Convert Timestamps to seconds since the epoch, calculate average, and convert back to Timestamp
        if first_spikes:
            avg_first_spike_seconds = sum((spike.timestamp() for spike in first_spikes)) / len(first_spikes)
            first_sync_time = pd.to_datetime(avg_first_spike_seconds, unit='s')
        else:
            first_sync_time = pd.to_datetime(first_sync_marker[1])

        if second_spikes:
            avg_second_spike_seconds = sum((spike.timestamp() for spike in second_spikes)) / len(second_spikes)
            second_sync_time = pd.to_datetime(avg_second_spike_seconds, unit='s')
        else:
            second_sync_time = pd.to_datetime(second_sync_marker[1])

        # Calculate the time shift for synchronization
        time_shift = second_sync_time - pd.to_datetime(second_sync_marker[1])

        synchronized_markers = self.adjust_marker_timestamps(event_markers, time_shift)

        return synchronized_markers, {'first_spikes': first_spikes, 'second_spikes': second_spikes}

    def detect_spikes_for_sync_events(self, acc_data, sync_markers, spike_threshold=1.0, min_spike_interval=1):
        """
        Detect spikes for each SYNC event within its respective window and ensure the spacing
        between these spikes matches the duration between the SYNC events.

        :param acc_data: DataFrame containing accelerometer data with a DateTime index.
        :param sync_markers: List of SYNC event markers.
        :param spike_threshold: The threshold for identifying a significant spike in the accelerometer data.
        :param min_spike_interval: The minimum interval (in seconds) between consecutive spikes.
        :return: Two lists of timestamps where significant spikes were detected for each SYNC event.
        """

        def detect_spikes_in_window(window_start, window_end):
            window_data = acc_data[(acc_data.index >= window_start) & (acc_data.index <= window_end)]
            spikes = []
            if not window_data.empty:
                acc_magnitude = np.sqrt((window_data ** 2).sum(axis=1))
                last_spike_time = None
                for time, magnitude in acc_magnitude.items():
                    if magnitude > spike_threshold and (
                            last_spike_time is None or (time - last_spike_time).total_seconds() > min_spike_interval):
                        spikes.append(time)
                        last_spike_time = time
                return spikes

        first_sync_marker, second_sync_marker = sync_markers[:2]
        first_sync_window = first_sync_marker[3].total_seconds()
        second_sync_window = second_sync_marker[3].total_seconds()

        # Detect spikes for each SYNC event within its respective window
        first_window_start = first_sync_marker[1] - pd.Timedelta(seconds=first_sync_window / 2)
        first_window_end = first_sync_marker[1] + pd.Timedelta(seconds=first_sync_window / 2)
        second_window_start = second_sync_marker[1] - pd.Timedelta(seconds=second_sync_window / 2)
        second_window_end = second_sync_marker[1] + pd.Timedelta(seconds=second_sync_window / 2)

        spikes_first_sync = detect_spikes_in_window(first_window_start, first_window_end)
        spikes_second_sync = detect_spikes_in_window(second_window_start, second_window_end)

        return spikes_first_sync, spikes_second_sync




    def detect_spikes_in_window(self, acc_data, target_time, window=10, spike_threshold=100.0, min_spike_interval=1):
        """
        Detect significant spikes in accelerometer data within a specified window around the target time.

        :param acc_data: DataFrame containing accelerometer data with a DateTime index.
        :param target_time: The central time around which to look for spikes.
        :param window: The duration (in seconds) of the window to examine around the target time.
        :param spike_threshold: The threshold for identifying a significant spike in the accelerometer data.
        :param min_spike_interval: The minimum interval (in seconds) between consecutive spikes.
        :return: List of timestamps where significant spikes were detected.
        """
        window_start = target_time - pd.Timedelta(seconds=window / 2)
        window_end = target_time + pd.Timedelta(seconds=window / 2)
        window_data = acc_data[(acc_data.index >= window_start) & (acc_data.index <= window_end)]

        spikes = []
        if not window_data.empty:
            # Compute the magnitude of accelerometer data
            acc_magnitude = np.sqrt((window_data ** 2).sum(axis=1))

            # Detect spikes
            last_spike_time = None
            for time, magnitude in acc_magnitude.items():  # Use items() instead of iteritems()
                if magnitude > spike_threshold and (
                        last_spike_time is None or (time - last_spike_time).total_seconds() > min_spike_interval):
                    spikes.append(time)
                    last_spike_time = time

        return spikes


    def calculate_time_shifts(self, original_markers, detected_sync_times):
        """
        Calculate time shifts based on the original and detected SYNC event times.

        :param original_markers: List of original event markers.
        :param detected_sync_times: Dictionary of detected SYNC times.
        :return: Dictionary of time shifts for SYNC events.
        """
        time_shifts = {}
        for marker, marker_time, category, duration in original_markers:
            if category == 'Sync Events':
                if marker_time in detected_sync_times:
                    detected_time = detected_sync_times[marker_time]
                    time_shifts[marker_time] = detected_time - marker_time
        return time_shifts

    def adjust_marker_timestamps(self, markers, first_sync_shift):
        """
        Adjust the timestamps of all markers based on the calculated time shift from the first SYNC event.

        :param markers: List of original event markers.
        :param first_sync_shift: Time shift calculated from the first SYNC event.
        :return: List of adjusted markers.
        """
        adjusted_markers = []
        for marker, marker_time, category, duration in markers:
            # Apply the first sync shift to all markers
            adjusted_time = marker_time + first_sync_shift
            adjusted_markers.append((marker, adjusted_time, category, duration))

        return adjusted_markers

    def load_event_logs_and_markers(self, logs_directory):
        event_logs = {}
        event_markers = {}
        time_difference = timedelta(minutes=0)

        for file in os.listdir(logs_directory):
            if file.endswith('_events.csv'):

                if 'a4880b' in file:
                    time_difference = timedelta(minutes=0)

                user_id, muse_id, empatica_id, _ = file.split('_')
                df = pd.read_csv(os.path.join(logs_directory, file))

                # Converting and adjusting timestamps
                df['Start'] = pd.to_datetime(df['Start'], unit='s') + time_difference
                df['End'] = pd.to_datetime(df['End'], unit='s') + time_difference

                stream_id = f"{user_id}_{muse_id}_{empatica_id}"
                event_logs[stream_id] = (df['Start'].min(), df['End'].max())

                # Process each event marker
                for _, row in df.iterrows():
                    event_marker = row['Marker']
                    event_time = row['Start']
                    event_duration = row['End'] - event_time
                    if stream_id not in event_markers:
                        event_markers[stream_id] = []
                    if row['Category'] == 'Yoga Poses':
                        event_time = event_time - timedelta(seconds=120)
                    event_markers[stream_id].append((event_marker, event_time, row['Category'], event_duration))

        return event_logs, event_markers

    def add_event_markers(self, data, markers, eeg_start_time):
        """
        Annotate the MNE raw data or DataFrame with event markers.
        """
        onsets = []
        durations = []
        descriptions = []

        for marker, marker_time, category, duration in markers:
            marker_time = pd.to_datetime(marker_time)
            onset = (marker_time - eeg_start_time).total_seconds()
            print("Marker time:", marker_time, "EEG start time:", eeg_start_time, "Onset (s):", onset)

            if isinstance(data, pd.DataFrame):
                # Use DataFrame index for time checks
                if marker_time in data.index:
                    onsets.append(onset)
                    durations.append(0)  # Update this if you have duration info
                    descriptions.append(marker)
            elif hasattr(data, 'times'):
                # Use MNE Raw object's times attribute
                if onset >= 0 and onset <= data.times[-1]:
                    onsets.append(onset)
                    durations.append(0)  # Update this if you have duration info
                    descriptions.append(marker)

        if isinstance(data, mne.io.Raw):
            # Create an Annotations object for MNE Raw
            annotations = mne.Annotations(onsets, durations, descriptions)
            data.set_annotations(annotations)

        return data

    def trim_data_to_events(self, data, start_time, end_time):
        if isinstance(data, mne.io.RawArray) or isinstance(data, mne.io.BaseRaw):
            return data
        elif isinstance(data, pd.DataFrame):
            # Convert index to DatetimeIndex if necessary
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, errors='coerce')

            # Check for any NaT (Not-a-Time) values in the index after conversion
            if data.index.isnull().any():
                raise ValueError("Index conversion to DatetimeIndex resulted in NaT values")

            # Trim the DataFrame
            filtered_data = data[(data.index >= start_time) & (data.index <= end_time)]

            # Check if the resulting DataFrame is empty
            if filtered_data.empty:
                print("Warning: The trimmed DataFrame is empty. Check start_time and end_time.")

            return filtered_data
        else:
            raise TypeError("Data must be either MNE Raw object or Pandas DataFrame")

    def _load_mne_raw(self, df, sfreq):
        df = self.handle_nans(df)
        data = df.values.T
        info = mne.create_info(ch_names=df.columns.tolist(), sfreq=sfreq, ch_types='eeg')

        raw = mne.io.RawArray(data, info)
        raw.drop_channels('R_AUX')
        return raw

    def plot_raw_data(self, raw_data, title='Raw EEG Data'):
        # Create a plot of the raw MNE data
        scalings = 'auto'  # Scale data to automatically determine suitable scaling factors
        raw_data.plot(scalings=scalings, title=title, show=True)

    def handle_nans(self, df):
        # Interpolate NaNs using linear interpolation
        df = df.interpolate(method='linear').fillna(0)
        return df




