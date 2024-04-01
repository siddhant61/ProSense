import glob
import os
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import LoadData

user_id_mapping = pd.read_csv('D:/Study Data/user_mapping.csv')

def flatten_eeg_features(eeg_features_dict, session, log_map):
    dfs = []
    ch_names = ["AF7", "AF8", "TP9", "TP10"]

    for stream_id, features in eeg_features_dict.items():
        data = []

        # Determine the number of epochs based on the length of one of the features
        total_epochs = features['total_epochs']

        for i in range(total_epochs):
            epoch_data = {'stream_id': stream_id, 'epoch': i}

            # Flatten power_band_ratios, psd_features, and statistical_features
            for feature_name in ['power_band_ratios', 'psd_features', 'statistical_features']:
                for channel_name, channel_data in features.get(feature_name, {})[i].items():
                    for metric, value in channel_data.items():
                        key = f"{channel_name}_{feature_name}_{metric}"
                        epoch_data[key] = value

            # Handle tfr_features
            if 'tfr_features' in features and i < features['tfr_features'].data.shape[2]:
                tfr_data_epoch = features['tfr_features'].data[:, :, i]
                tfr_data_epoch_flat = tfr_data_epoch.flatten()
                for j, value in enumerate(tfr_data_epoch_flat):
                    epoch_data[f'tfr_features_{j}'] = value

            # Handle spectral_entropy, coherence_features, and epoch_features
            for feature_name in ['spectral_entropy', 'coherence_features', 'epoch_features']:
                if feature_name in features:
                    if feature_name == 'epoch_features':
                        for metric, values in features[feature_name].items():
                            for idx, channel_name in enumerate(ch_names):
                                epoch_data[f"{channel_name}_{metric}"] = values[i][idx]
                    else:
                        for channel_name, channel_value in features[feature_name][i].items():
                            epoch_data[f"{feature_name}_{channel_name}"] = channel_value

            epoch_data['session'] = session
            epoch_data['user_id'] = get_user_id_for_stream(stream_id, log_map)

            data.append(epoch_data)
        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    return df


def flatten_gsr_features(gsr_features, session, log_map):
    gsr_flattened_data = []

    for stream_id, epochs in gsr_features.items():
        for i, epoch_features in enumerate(epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': i}

            # Add scalar features directly
            for feature_name, value in epoch_features.items():
                if not isinstance(value, pd.Series):
                    flattened_features[feature_name] = value

                # Handle Series data (e.g., GSR Variance)
                if feature_name == 'GSR Variance':
                    for j, var_value in enumerate(value):
                        flattened_features[f'GSR_Variance_{j}'] = var_value

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            gsr_flattened_data.append(flattened_features)

    return pd.DataFrame(gsr_flattened_data)


def flatten_ppg_features(ppg_features, session, log_map):
    ppg_flattened_data = []

    for stream_id, feature_categories in ppg_features.items():
        num_epochs = len(feature_categories['hr_features'])

        for epoch in range(num_epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': epoch}

            # Iterate over each feature category
            for category_name, category_data in feature_categories.items():
                # Flatten features for each channel within the epoch
                for channel, channel_features in category_data[epoch].items():
                    for feature_name, value in channel_features.items():
                        flattened_features[f"{category_name}_{channel}_{feature_name}"] = value

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            ppg_flattened_data.append(flattened_features)

    return pd.DataFrame(ppg_flattened_data)


def flatten_temp_features(temp_features, session, log_map):
    temp_flattened_data = []

    for stream_id, epochs in temp_features.items():
        for i, epoch_features in enumerate(epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': i}

            # Add features from the epoch
            for feature_name, value in epoch_features.items():
                flattened_features[feature_name] = value

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            temp_flattened_data.append(flattened_features)

    return pd.DataFrame(temp_flattened_data)


def flatten_acc_features(acc_features, session, log_map):
    acc_flattened_data = []

    for stream_id, feature_categories in acc_features.items():
        num_epochs = len(feature_categories['time_features'])
        for epoch in range(num_epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': epoch}

            # Iterate over each feature category
            for category_name, category_data in feature_categories.items():
                # Flatten features for each channel within the epoch
                for channel, channel_features in category_data[epoch].items():
                    for feature_name, value in channel_features.items():
                        flattened_features[f"{category_name}_{channel}_{feature_name}"] = value

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            acc_flattened_data.append(flattened_features)

    return pd.DataFrame(acc_flattened_data)


def flatten_gyro_features(gyro_features, session, log_map):
    gyro_flattened_data = []

    for stream_id, feature_categories in gyro_features.items():
        num_epochs = len(feature_categories['time_features'])
        for epoch in range(num_epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': epoch}

            # Iterate over each feature category
            for category_name, category_data in feature_categories.items():
                # Flatten features for each channel within the epoch
                for channel, channel_features in category_data[epoch].items():
                    for feature_name, value in channel_features.items():
                        flattened_features[f"{category_name}_{channel}_{feature_name}"] = value

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            gyro_flattened_data.append(flattened_features)

    return pd.DataFrame(gyro_flattened_data)


def flatten_bvp_features(bvp_features, session, log_map):
    bvp_flattened_data = []

    for stream_id, feature_categories in bvp_features.items():
        num_epochs = len(feature_categories[list(feature_categories.keys())[0]])

        for epoch in range(num_epochs):
            flattened_features = {'stream_id': stream_id, 'epoch': epoch}

            for category_name, category_data in feature_categories.items():
                feature_values = category_data[epoch]
                if isinstance(feature_values, dict):
                    for key, value in feature_values.items():
                        flattened_features[f"{category_name}_{key}"] = value
                else:
                    flattened_features[category_name] = feature_values

            flattened_features['session'] = session
            flattened_features['user_id'] = get_user_id_for_stream(stream_id, log_map)
            bvp_flattened_data.append(flattened_features)

    return pd.DataFrame(bvp_flattened_data)


def merge_features_with_events(feature_df, events_df):
    # If feature_df is a MultiIndex DataFrame, reset the index to make 'stream_id', 'epoch', and 'user_id' columns
    if isinstance(feature_df.index, pd.MultiIndex):
        feature_df = feature_df.reset_index()

    # Same for events_df, ensure it's not a MultiIndex for easier manipulation
    if isinstance(events_df.index, pd.MultiIndex):
        events_df = events_df.reset_index()

    # Add columns for event details if they don't already exist
    event_columns = ['event_name', 'event_type', 'event_duration', 'metadata']
    for col in event_columns:
        if col not in feature_df.columns:
            feature_df[col] = None

    # Iterate through each unique stream_id in feature_df
    for stream_id in feature_df['stream_id'].unique():
        # Select features and events corresponding to the current stream_id
        stream_features = feature_df[feature_df['stream_id'] == stream_id]
        stream_events = events_df[events_df['stream_id'] == stream_id].sort_values(by='event_id')

        # Calculate and assign event details for each feature row
        for _, event in stream_events.iterrows():
            # Determine the range of epochs affected by the current event
            event_start_epoch = event['event_id']
            event_end_epoch = event_start_epoch + int(
                event['timedelta'].total_seconds() / 5)  # Assuming 5 seconds per epoch

            # Update the corresponding features with the event's details
            for epoch in range(event_start_epoch, event_end_epoch + 1):
                condition = (feature_df['stream_id'] == stream_id) & (feature_df['epoch'] == epoch)
                feature_df.loc[condition, 'event_name'] = event['event_name']
                feature_df.loc[condition, 'event_type'] = event['event_type']
                feature_df.loc[condition, 'event_duration'] = event['timedelta'].total_seconds()
                feature_df.loc[condition, 'metadata'] = str(event['Metadata'])

    # Ensure 'stream_id', 'epoch', and 'user_id' are kept in the DataFrame
    # No need to set them as index again if you need them as columns

    return feature_df


# Function to check if a stream_id belongs to a log_key and return the corresponding user_id
def get_user_id_for_stream(stream_id, log_mapping):
    sensor_id = stream_id.split('_')[0].lower()
    if 'muses-' in sensor_id:
        sensor_id = sensor_id.replace('muses-', '')
    for log_key in log_mapping.values():
        if sensor_id in log_key:
            original_user_id = log_key.split('_')[0]  # Extracting the user_id from the log_key
            # Return the pseudonym for the user_id
            return user_id_mapping.get(original_user_id, None)
    return None


if __name__ == '__main__':

    loader = LoadData()

    initial_path = "D:/Study Data/set_1/session_1/datasets/ProcessedData/Features/"
    inital_log_path = "D:/Study Data/set_1/session_1/datasets/ProcessedData/Events/events.pkl"
    initial_log_files = "D:/Study Data/set_1/session_1/logs/ProcessedLogs/"

    initial_log_map = "D:/Study Data/set_1/session_1/datasets/ProcessedData/Events/log_map.pkl"

    sessions = 5

    all_sessions_data = {
        'eeg': [],
        'gsr': [],
        'ppg': [],
        'temp': [],
        'acc': [],
        'gyro': [],
        'bvp': []
    }

    for i in range(1, sessions + 1):
        folder_path = initial_path.replace('session_1', f'session_{i}')

        log_path = inital_log_path.replace('session_1', f'session_{i}')

        log_files = initial_log_files.replace('session_1', f'session_{i}')

        log_map_path = initial_log_map.replace('session_1', f'session_{i}')

        log_map = loader.load_pkl_dataset(log_map_path)

        output_path = Path(f"D:/Study Data/bz_fk/session_{i}/output/")
        output_path.mkdir(parents=True, exist_ok=True)

        events = loader.load_pkl_dataset(log_path)

        # Convert the dictionary to a pandas DataFrame
        df = pd.concat({k: pd.DataFrame(v, columns=['event_name', 'timestamp', 'event_type', 'timedelta']) for k, v in events.items()}, names=['stream_id', 'epoch'])

        # Convert the 'Timestamp' and 'Timedelta' columns to their appropriate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timedelta'] = pd.to_timedelta(df['timedelta'])


        for file in os.listdir(log_files):
            if file.endswith('_events.csv'):
                log_df = pd.read_csv(os.path.join(log_files, file))
                muse = file.split('_')[1].replace('m','')
                emp = file.split('_')[2].replace('e', '')
                for df_id in df.index.get_level_values('stream_id').unique():
                    if muse in df_id.lower() or emp in df_id.lower():
                        for index, row in df.loc[df_id].iterrows():
                            if row['event_name'] in log_df['Marker'].values:
                                df.loc[(df_id, index), 'Metadata'] = \
                                log_df.loc[log_df['Marker'] == row['event_name'], 'Metadata'].values[0]

        # Reset the index
        df_reset = df.reset_index()

        # Rename the column
        df_reset = df_reset.rename(columns={'epoch': 'event_id'})

        # Set the index again
        df = df_reset.set_index(['stream_id', 'event_id'])

        df.to_csv(f'{output_path}/events.csv')
        print(df.head())

        eeg_features = pd.read_pickle(f'{folder_path}/eeg_features.pkl')
        eeg_flattened_df = flatten_eeg_features(eeg_features, i, log_map)
        eeg_flattened_df = eeg_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(eeg_flattened_df.head())
        merged_eeg = merge_features_with_events(eeg_flattened_df, df)
        all_sessions_data['eeg'].append(merged_eeg)
        merged_eeg.to_csv(f'{output_path}/eeg_features.csv', index=False)


        gsr_features = pd.read_pickle(f'{folder_path}/gsr_features.pkl')
        gsr_flattened_df = flatten_gsr_features(gsr_features, i, log_map)
        gsr_flattened_df = gsr_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(gsr_flattened_df.head())
        merged_gsr = merge_features_with_events(gsr_flattened_df, df)
        all_sessions_data['gsr'].append(merged_gsr)
        merged_gsr.to_csv(f'{output_path}/gsr_features.csv', index=False)


        ppg_features = pd.read_pickle(f'{folder_path}/ppg_features.pkl')
        ppg_flattened_df = flatten_ppg_features(ppg_features, i, log_map)
        ppg_flattened_df = ppg_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(ppg_flattened_df.head())
        merged_ppg = merge_features_with_events(ppg_flattened_df, df)
        all_sessions_data['ppg'].append(merged_ppg)
        merged_ppg.to_csv(f'{output_path}/ppg_features.csv', index=False)


        temp_features = pd.read_pickle(f'{folder_path}/temp_features.pkl')
        temp_flattened_df = flatten_temp_features(temp_features, i, log_map)
        temp_flattened_df = temp_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(temp_flattened_df.head())
        merged_temp = merge_features_with_events(temp_flattened_df, df)
        all_sessions_data['temp'].append(merged_temp)
        merged_temp.to_csv(f'{output_path}/temp_features.csv', index=False)


        acc_features = pd.read_pickle(f'{folder_path}/acc_features.pkl')
        acc_flattened_df = flatten_acc_features(acc_features, i, log_map)
        acc_flattened_df = acc_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(acc_flattened_df.head())
        merged_acc = merge_features_with_events(acc_flattened_df, df)
        all_sessions_data['acc'].append(merged_acc)
        merged_acc.to_csv(f'{output_path}/acc_features.csv', index=False)


        gyro_features = pd.read_pickle(f'{folder_path}/gyro_features.pkl')
        gyro_flattened_df = flatten_gyro_features(gyro_features, i, log_map)
        gyro_flattened_df = gyro_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(gyro_flattened_df.head())
        merged_gyro = merge_features_with_events(gyro_flattened_df, df)
        all_sessions_data['gyro'].append(merged_gyro)
        merged_gyro.to_csv(f'{output_path}/gyro_features.csv', index=False)


        bvp_features = pd.read_pickle(f'{folder_path}/bvp_features.pkl')
        bvp_flattened_df = flatten_bvp_features(bvp_features, i, log_map)
        bvp_flattened_df = bvp_flattened_df.set_index(['stream_id', 'epoch', 'user_id'])
        print(bvp_flattened_df.head())
        merged_bvp = merge_features_with_events(bvp_flattened_df, df)
        all_sessions_data['bvp'].append(merged_bvp)
        merged_bvp.to_csv(f'{output_path}/bvp_features.csv', index=False)

    for feature_type in all_sessions_data:
        combined_df = pd.concat(all_sessions_data[feature_type], ignore_index=True)
        combined_df.to_csv(f'D:/Study Data/set_1/{feature_type}_all_sessions.csv', index=False)

