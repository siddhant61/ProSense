import os
from pathlib import Path
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import LoadData


# Merging each feature dataset with the events dataset
def merge_features_with_events(feature_df, event_df):
    # Merging based on 'stream_id' and 'epoch'
    merged_df = pd.merge(feature_df, event_df, on=['stream_id', 'epoch'], how='inner')
    return merged_df


def calculate_correlation(features_df, event_indicators_df, threshold=0.3):
    """
    Calculate correlations between numeric feature columns and event binary indicators.
    """
    # Extract numeric feature columns
    feature_columns = features_df.select_dtypes(include=[np.number]).columns

    # Initialize a DataFrame to store the correlation results
    correlation_results = pd.DataFrame()

    # Calculate the correlation for each event indicator
    for event_indicator in event_indicators_df.columns:
        correlation_series = features_df[feature_columns].corrwith(event_indicators_df[event_indicator])
        correlation_series = correlation_series[abs(correlation_series) >= threshold].dropna()
        if not correlation_series.empty:
            correlation_results[event_indicator] = correlation_series

    return correlation_results

def visualize_correlation(correlation_df, title="Correlation Matrix"):
    """
    Visualize the correlation data as a heatmap.
    """
    if correlation_df.empty:
        print("No correlation data to visualize.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def create_binary_indicators(df, column):
    for unique_value in df[column].unique():
        indicator_name = f"{column}_{unique_value}"
        df[indicator_name] = (df[column] == unique_value).astype(int)
    return df


# To calculate correlation between features and events, we need to encode events as numerical values
def encode_events(feature_event_df, event_column):
    encoded = pd.factorize(feature_event_df[event_column])[0]
    return pd.concat([feature_event_df, pd.Series(encoded, name=event_column + '_encoded')], axis=1)


# Function to check if a stream_id belongs to a log_key and return the corresponding user_id
def get_user_id_for_stream(stream_id, log_mapping):
    sensor_id = stream_id.split('_')[0].lower()
    if 'muses-' in sensor_id:
        sensor_id = sensor_id.replace('muses-', '')
    for log_key in log_mapping.values():
        if sensor_id in log_key:
            return log_key.split('_')[0]  # Extracting the user_id from the log_key
    return None



# Function to group datasets by user_id based on log_mapping
def group_datasets_by_user_id(datasets, log_mapping):
    grouped_datasets = {}
    for name, df in datasets.items():

        # Map each record to the corresponding user_id
        df['user_id'] = df['stream_id'].apply(lambda x: get_user_id_for_stream(x, log_mapping))

        # Group by user_id
        for user_id in df['user_id'].unique():
            grouped_key = f"{user_id}_{name}"
            grouped_datasets[grouped_key] = df[df['user_id'] == user_id]
    return grouped_datasets

# Function to calculate correlations between different stream types
def calculate_cross_stream_correlation(grouped_datasets, events_df, threshold=0.3):
    correlation_results = {}
    for key1, df1 in grouped_datasets.items():
        for key2, df2 in grouped_datasets.items():
            if key1.split('_')[0] == key2.split('_')[0] and key1 != key2:
                df1_with_events = merge_features_with_events(df1, events_df)
                df2_with_events = merge_features_with_events(df2, events_df)

                # Select only numeric columns for correlation calculation
                df1_numeric = df1_with_events.select_dtypes(include=[np.number])
                df2_numeric = df2_with_events.select_dtypes(include=[np.number])

                merged_df = pd.merge(df1_numeric, df2_numeric, on=['epoch'], how='inner')
                correlation = calculate_correlation(merged_df, merged_df.columns[-1], threshold)
                correlation_results[f"{key1}_{key2}"] = correlation
    return correlation_results


def plot_individual_user_sensor_events(data, feature, user_id, selected_event_names=None):
    """
    Plots a continuous line for all data points for each sensor and marks events with vertical dotted lines,
    common to both sensor data streams.

    :param data: DataFrame containing the data.
    :param feature: The feature column to be plotted.
    :param user_id: User ID for which to plot the data.
    :param selected_event_names: (Optional) List of specific event names to mark with vertical lines.
    """
    user_data = data[data['user_id'] == user_id].copy()
    user_data.sort_values(by='timestamp', inplace=True)
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plotting data for each sensor
    for sensor in user_data['stream_id'].unique():
        sensor_data = user_data[user_data['stream_id'] == sensor]
        plt.plot(sensor_data['timestamp'], sensor_data[feature], label=f'Sensor {sensor}', linestyle='-')

    # Marking the events with vertical lines if specified
    if selected_event_names is None:
        selected_event_names = user_data['event_name'].unique()

    for event_name in selected_event_names:
        event_occurrences = user_data[user_data['event_name'] == event_name]
        for _, event_row in event_occurrences.iterrows():
            plt.axvline(x=event_row['timestamp'], color='grey', alpha=0.5, linestyle='--')
            plt.text(event_row['timestamp'], plt.ylim()[1], event_name, fontsize=9, verticalalignment='top', rotation=45)

    title = f"{feature} Over Time for User {user_id}"
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(feature)
    ax.legend(loc='upper left')
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    return fig, title


def plot_combined_user_sensor_events_with_markers(data, feature, user_id, selected_event_types=None):
    """
    Creates a plot for a specific user, with distinct legend entries for event types (colors) and sensor IDs (markers).
    Returns the plot object and title for external saving.

    :param data: DataFrame containing the data.
    :param feature: The feature column to be plotted.
    :param user_id: User ID for which to plot the data.
    :param selected_event_types: (Optional) List of specific event types to include in the plot.
    """
    # Filter data for the specified user
    user_data = data[data['user_id'] == user_id]

    # Ensure data is sorted by timestamp
    user_data.sort_values(by='timestamp', inplace=True)

    # Convert string timestamps to datetime
    user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])

    # Defining colors for event types
    unique_event_types = user_data['event_type'].unique()
    colors = sns.color_palette("tab10", len(unique_event_types))
    event_type_colors = dict(zip(unique_event_types, colors))

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(15, 6))

    # Markers for each sensor
    markers = ['o', 's', '^', 'x', '+']  # Add more markers if there are more than 5 sensors

    # Plotting line segments for each sensor
    for sensor_idx, sensor in enumerate(user_data['stream_id'].unique()):
        sensor_data = user_data[user_data['stream_id'] == sensor]
        marker = markers[sensor_idx % len(markers)]

        for i in range(len(sensor_data) - 1):
            start = sensor_data.iloc[i]
            end = sensor_data.iloc[i + 1]
            color = event_type_colors.get(start['event_type'], 'grey')
            ax.plot([start['timestamp'], end['timestamp']], [start[feature], end[feature]], color=color, marker=marker)

    # Construct a title for the plot
    title = f"{feature} Over Time for User {user_id}"

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(feature)

    # Creating custom legends
    event_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in event_type_colors.values()]
    event_legend = ax.legend(event_lines, unique_event_types, title="Event Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    sensor_lines = [plt.Line2D([0], [0], color='black', lw=2, marker=marker) for marker in markers[:len(user_data['stream_id'].unique())]]
    sensor_legend = ax.legend(sensor_lines, user_data['stream_id'].unique(), title="Sensors", bbox_to_anchor=(1.05, 0.2), loc='upper left')

    ax.add_artist(event_legend)

    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.tight_layout()

    return fig, title

def plot_features_for_all_users(merged_data, non_feature_columns,type, save_path , selected_events=None, selected_event_types=None):
    """
    Plots all features for each user in the merged data.

    :param merged_data: DataFrame containing the merged data.
    :param non_feature_columns: List of columns that are not features.
    :param selected_events: (Optional) List of specific event names to include in the plot.
    :param selected_event_types: (Optional) List of specific event types to include in the plot.
    """
    feature_columns = [col for col in merged_data.columns if col not in non_feature_columns]
    users = merged_data['user_id'].unique()

    for user_id in users:
        for feature in feature_columns:
            if selected_event_types:
                fig, title = plot_combined_user_sensor_events_with_markers(merged_data, feature, user_id, selected_event_types)
                save_figures([fig], [title], f'{type}', save_path)
            elif selected_events:
                fig, title = plot_individual_user_sensor_events(merged_data, feature, user_id, selected_events)
                save_figures([fig], [title], f'{type}', save_path)
            else:
                # Default plot if no specific events or event types are provided
                fig, title = plot_individual_user_sensor_events(merged_data, feature, user_id)
                save_figures([fig], [title], f'{type}', save_path)


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



if __name__ == '__main__':

    loader = LoadData()

    initial_output_path = "D:/Study Data/pb_kk"
    initial_log_map = "D:/Study Data/pb_kk/session_1/datasets/ProcessedData/Events/log_map.pkl"
    sessions = 5

    for i in range(1, sessions + 1):
        folder_path = initial_output_path.replace('session_1', f'session_{i}')

        initial_log_map = initial_log_map.replace('session_1', f'session_{i}')

        log_map = loader.load_pkl_dataset(initial_log_map)
        eeg_features_path = f'{folder_path}/eeg_all_sessions.csv'
        gsr_features_path = f'{folder_path}/gsr_all_sessions.csv'
        gyro_features_path = f'{folder_path}/gyro_all_sessions.csv'
        ppg_features_path = f'{folder_path}/ppg_all_sessions.csv'
        temp_features_path = f'{folder_path}/temp_all_sessions.csv'
        acc_features_path = f'{folder_path}/acc_all_sessions.csv'
        bvp_features_path = f'{folder_path}/bvp_all_sessions.csv'

        # Loading the datasets
        eeg_features = pd.read_csv(eeg_features_path)
        gsr_features = pd.read_csv(gsr_features_path)
        gyro_features = pd.read_csv(gyro_features_path)
        ppg_features = pd.read_csv(ppg_features_path)
        temp_features = pd.read_csv(temp_features_path)
        acc_features = pd.read_csv(acc_features_path)
        bvp_features = pd.read_csv(bvp_features_path)

        selected_events = [
            'start_BRUMS', 'parity_set_1', 'dualt_acc_1', 'individualization_acc_1',
            'final_test_set_1_instr', 'final_test_instr', 'childs_pose_1',
            'low_lunge_1', 'cobra_pose_1', 'low_lunge_2', 'cobra_pose_2',
            'tree_pose_1', 'high_lunge_1', 'childs_pose_4', 'savasana_1', 'closing_chant_1'
        ]

        # plot_individual_user_sensor_events(merged_acc, 'time_features_x_mean', 'kk', selected_events)

        selected_types = ['Instruction Events', 'Sync Events', 'Zero Score Trials', 'Accuracy Events', 'Yoga Poses']

        # plot_combined_user_sensor_events_with_markers(merged_acc, 'time_features_x_mean', 'kk')

        non_feature_cols = ['stream_id', 'epoch', 'user_id', 'event_name', 'timestamp', 'event_type', 'timedelta',
                            'Metadata']

        plot_features_for_all_users(eeg_features, non_feature_cols, 'EEG', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(gsr_features, non_feature_cols, 'GSR', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(gyro_features, non_feature_cols, 'GYRO', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(ppg_features, non_feature_cols, 'PPG', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(temp_features, non_feature_cols, 'TEMP', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(acc_features, non_feature_cols, 'ACC', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)
        plot_features_for_all_users(bvp_features, non_feature_cols, 'BVP', f'{folder_path}/ProcessedData/Plots', selected_events=selected_events)

        # Create binary indicators for event types and event names
        merged_eeg_with_types = create_binary_indicators(eeg_features, 'event_type')
        merged_eeg_with_names = create_binary_indicators(eeg_features, 'event_name')

        # Prepare columns for correlation
        event_type_indicators = merged_eeg_with_types.filter(like='event_type_')
        event_name_indicators = merged_eeg_with_names.filter(like='event_name_')

        # Calculate correlations
        correlation_results_event_types = calculate_correlation(merged_eeg_with_types, event_type_indicators,
                                                                threshold=0.3)
        correlation_results_event_names = calculate_correlation(merged_eeg_with_names, event_name_indicators,
                                                                threshold=0.3)

        # Visualization
        if not correlation_results_event_types.empty:
            visualize_correlation(correlation_results_event_types, "EEG Features and Event Types Correlation")
        if not correlation_results_event_names.empty:
            visualize_correlation(correlation_results_event_names, "EEG Features and Event Names Correlation")
        #
        # grouped_datasets = group_datasets_by_user_id({
        #     "EEG": merged_eeg,
        #     "GSR": merged_gsr,
        #     "GYRO": merged_gyro,
        #     "PPG": merged_ppg,
        #     "TEMP": merged_temp,
        #     "ACC": merged_acc,
        #     "BVP": merged_bvp
        # }, log_map)
        #
        # print(grouped_datasets.keys())
        #
        # cross_stream_correlations = calculate_cross_stream_correlation(grouped_datasets, events, 0.3)
        # if cross_stream_correlations:
        #     for key, corr_matrix in cross_stream_correlations.items():
        #         if not corr_matrix.empty:
        #             visualize_correlation(corr_matrix, f"Cross-stream Correlation: {key}")






#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def compute_timestamps(data_df):
#     """
#     Computes timestamps by multiplying the epoch value (row index) by 5.
#     """
#     data_df['timestamp'] = data_df.index * 5
#     data_df['timestamp'] = pd.to_timedelta(data_df['timestamp'], unit='s')
#     return data_df
#
# def average_data_during_event(data_df, start_time, end_time):
#     """
#     Calculate the average of the data between start_time and end_time.
#     """
#     relevant_data = data_df[(data_df['timestamp'] >= start_time) & (data_df['timestamp'] <= end_time)]
#     return relevant_data.mean()
#
# def correlate_events_with_data(data_df, events_df, data_stream_id):
#     """
#     Correlates events with data, calculating the average data during each event.
#     """
#     correlations = []
#     filtered_events = events_df[events_df['ID'].str.contains(data_stream_id)]
#     for _, event in filtered_events.iterrows():
#         start_time = pd.Timedelta(event['Timestamp'])
#         # Assuming end_time is the next event's start time or the last timestamp in data_df
#         end_time = filtered_events['Timestamp'].shift(-1).fillna(data_df['timestamp'].iloc[-1]).iloc[_]
#         end_time = pd.Timedelta(end_time)
#         avg_data = average_data_during_event(data_df, start_time, end_time)
#         correlations.append(avg_data)
#
#     return pd.DataFrame(correlations)
#
# # Load data
# events_df = pd.read_csv('D:/Study Data/pd_pn/session_1/events.csv')
# eeg_df = pd.read_csv('D:/Study Data/pd_pn/session_1/eeg_features.csv')
# bvp_df = pd.read_csv('D:/Study Data/pd_pn/session_1/bvp_features.csv')
# acc_df = pd.read_csv('D:/Study Data/pd_pn/session_1/acc_features.csv')
#
# import pandas as pd
#
# # Load EEG Features Data
# eeg_features_file = 'D:/Study Data/bz_fk/session_1/gyro_features.csv'  # Update the path to your EEG features file
# eeg_features_data = pd.read_csv(eeg_features_file)
#
# # Load Events Data
# events_file = 'D:/Study Data/bz_fk/session_1/events.csv'  # Update the path to your events file
# events_data = pd.read_csv(events_file)
#
# # Assigning epoch numbers based on the order of events in each file
# events_data['Calculated Epoch'] = events_data.groupby('ID').cumcount()
#
# # Renaming columns in the events data for merging
# events_data_renamed = events_data.rename(columns={'ID': 'stream_id', 'Calculated Epoch': 'epoch'})
#
# # Merging EEG features data with the events data
# merged_data = pd.merge(eeg_features_data, events_data_renamed, on=['stream_id', 'epoch'], how='inner')
#
# # Prepare binary indicators for each event type
# event_types = merged_data['Event Type'].unique()
# for event_type in event_types:
#     merged_data[event_type] = (merged_data['Event Type'] == event_type).astype(int)
# # Convert all EEG feature columns to numeric (if they are not already)
# eeg_features = merged_data.columns[2:-len(event_types)-4]  # Adjust indices based on your data
# for feature in eeg_features:
#     merged_data[feature] = pd.to_numeric(merged_data[feature], errors='coerce')
#
# # Prepare binary indicators for each event type
# event_types = merged_data['Event Type'].unique()
# for event_type in event_types:
#     merged_data[event_type] = (merged_data['Event Type'] == event_type).astype(int)
#
# # List to store correlation results
# correlation_data = []
#
# for feature in eeg_features:
#     correlations = {'GYRO Feature': feature}
#     for event_type in event_types:
#         # Calculate correlation
#         try:
#             correlation = merged_data[feature].corr(merged_data[event_type])
#             correlations[event_type] = correlation
#         except TypeError:
#             # If a TypeError occurred during correlation calculation
#             correlations[event_type] = 'Error'
#     # Add to the list
#     correlation_data.append(correlations)
#
# # Convert the list to a DataFrame
# correlation_results = pd.DataFrame(correlation_data)
#
# # # # # Filter out rows where any correlation is NaN or the feature is not relevant
# # correlation_results = correlation_results.dropna().query("BVP Feature: not in ['Unnamed: 1', 'Event Name']")
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Set a threshold for strong correlations (absolute value)
# correlation_threshold = 0.1
#
# # Convert the DataFrame to a correlation matrix and filter
# correlation_matrix = correlation_results.set_index('GYRO Feature')
# strong_correlations = correlation_matrix[correlation_matrix.abs() >= correlation_threshold]
#
# # Plotting the heatmap of strong correlations
# plt.figure(figsize=(12, 10))  # Adjust size as needed
# sns.heatmap(strong_correlations, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Strong Correlations between GYRO Features and Event Types')
# plt.xticks(rotation=45, ha='right')  # Rotate x labels for readability
# plt.yticks(rotation=0)  # Rotate y labels for readability
# plt.tight_layout()  # Adjust layout
# plt.show()
