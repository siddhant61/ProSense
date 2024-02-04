import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def load_datasets(folder_path):
    """
    Load all datasets from a given folder path.
    """
    dataset_paths = {
        'eeg': f'{folder_path}/eeg_all_sessions.csv',
        'gsr': f'{folder_path}/gsr_all_sessions.csv',
        'gyro': f'{folder_path}/gyro_all_sessions.csv',
        'ppg': f'{folder_path}/ppg_all_sessions.csv',
        'temp': f'{folder_path}/temp_all_sessions.csv',
        'acc': f'{folder_path}/acc_all_sessions.csv',
        'bvp': f'{folder_path}/bvp_all_sessions.csv'
    }
    datasets = {}
    for key, path in dataset_paths.items():
        if os.path.exists(path):
            datasets[key] = pd.read_csv(path)
    return datasets


def combine_datasets_by_stream_type(session_folders):
    """
    Combine datasets by stream type across multiple sessions or user-pairs.

    :param session_folders: List of folder paths for each session or user-pair.
    :return: Dictionary of combined datasets keyed by stream type.
    """
    combined_datasets = {}
    for folder in session_folders:
        session_datasets = load_datasets(folder)
        for stream_type, data in session_datasets.items():
            if stream_type in combined_datasets:
                combined_datasets[stream_type] = pd.concat([combined_datasets[stream_type], data], ignore_index=True)
            else:
                combined_datasets[stream_type] = data
    return combined_datasets

def average_data_across_users(data, non_feature_cols):
    """
    Averages data for each session and phase, excluding non-numeric columns.
    Includes the most frequent 'event_name' as a representative event for each group.

    :param data: DataFrame containing the data with 'session', 'event_type', and 'event_name' columns.
    :param non_feature_cols: List of columns that are not features and should not be averaged.
    :return: DataFrame with averaged data per session, phase, and a representative event name.
    """
    # Create the 'phase' column based on 'event_type'
    data['phase'] = data['event_type'].apply(lambda x: 'Yoga Phase' if x == 'Yoga Poses' else 'Cognitive Load Phase')

    # Group data and calculate mean for numeric features
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
    averaged_data = data.groupby(['epoch', 'session', 'phase'], as_index=False)[feature_cols].mean()

    # Get the most frequent 'event_name' within each group or a default value if the group is empty
    frequent_event = data.groupby(['session', 'phase'])['event_name'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'No Event'
    ).reset_index()

    # Merge the frequent event data with the averaged data
    averaged_data = pd.merge(averaged_data, frequent_event, on=['session', 'phase'])

    return averaged_data

def plot_single_feature_per_session(averaged_data, feature, phase_filter=None):
    """
    Plots a single feature for all users per session and returns the figure.
    """
    figures = []
    sessions = averaged_data['session'].unique()

    for session in sessions:
        session_data = averaged_data[averaged_data['session'] == session]
        if phase_filter:
            session_data = session_data[session_data['phase'] == phase_filter]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=session_data, x='epoch', y=feature, marker='o', ax=ax)
        ax.set_title(f'Session {session} - {feature}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(feature)
        figures.append(fig)

    return figures




def plot_avg_features_all_users_per_session(averaged_data, non_feature_columns, phase_filter=None):
    """
    Plots average features for all users per session for a specific phase or all phases.
    :param averaged_data: DataFrame containing the averaged data.
    :param non_feature_columns: List of columns that are not features.
    :param phase_filter: (Optional) Specific phase to filter for.
    """
    feature_columns = [col for col in averaged_data.columns if col not in non_feature_columns]
    sessions = averaged_data['session'].unique()

    for session in sessions:
        session_data = averaged_data[averaged_data['session'] == session]
        if phase_filter:
            session_data = session_data[session_data['phase'] == phase_filter]

        fig, axes = plt.subplots(len(feature_columns), 1, figsize=(10, 6 * len(feature_columns)))
        for i, feature in enumerate(feature_columns):
            sns.lineplot(data=session_data, x='epoch', y=feature, ax=axes[i], marker='o')
            title_phase = phase_filter if phase_filter else 'All Phases'
            axes[i].set_title(f'Session {session} - {title_phase} - Average {feature}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(feature)
        plt.tight_layout()
        plt.show()

def plot_avg_features_per_user_across_sessions(averaged_data, feature, phase_filter=None):
    """
    Plots average of a single feature for each user across all sessions.
    :param averaged_data: DataFrame containing the averaged data.
    :param feature: The feature to plot.
    :param phase_filter: (Optional) Specific phase to filter for.
    """
    users = averaged_data['user_id'].unique()

    for user_id in users:
        user_data = averaged_data[averaged_data['user_id'] == user_id]
        if phase_filter:
            user_data = user_data[user_data['phase'] == phase_filter]

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=user_data, x='session', y=feature, marker='o')
        title_phase = phase_filter if phase_filter else 'All Phases'
        plt.title(f'User {user_id} - {title_phase} - Average {feature} Across Sessions')
        plt.xlabel('Session')
        plt.ylabel(feature)
        plt.show()


def compare_avg_features_first_second_participants(averaged_data, feature, phase_filter=None):
    """
    Compares average of a single feature between first and second participants for each session.
    :param averaged_data: DataFrame containing the averaged data.
    :param feature: The feature to plot.
    :param phase_filter: (Optional) Specific phase to filter for.
    """
    sessions = averaged_data['session'].unique()
    participant_pairs = [('P1', 'P2'), ('P3', 'P4'), ('P5', 'P6'), ('P7', 'P8'), ('P9', 'P10')]

    for session in sessions:
        session_data = averaged_data[averaged_data['session'] == session]
        if phase_filter:
            session_data = session_data[session_data['phase'] == phase_filter]

        plt.figure(figsize=(10, 6))
        for pair in participant_pairs:
            first_participant_data = session_data[session_data['user_id'] == pair[0]]
            second_participant_data = session_data[session_data['user_id'] == pair[1]]
            sns.lineplot(data=first_participant_data, x='epoch', y=feature, marker='o',
                         label=f'{pair[0]} - Session {session}')
            sns.lineplot(data=second_participant_data, x='epoch', y=feature, marker='o',
                         label=f'{pair[1]} - Session {session}')

        title_phase = phase_filter if phase_filter else 'All Phases'
        plt.title(f'Session {session} - {title_phase} - Comparison of {feature}')
        plt.xlabel('Epoch')
        plt.ylabel(feature)
        plt.legend()
        plt.show()


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


def main():
    session_folders = ['D:/Study Data/bz_fk', 'D:/Study Data/jk_ka', 'D:/Study Data/pb_kk', 'D:/Study Data/pd_pn', 'D:/Study Data/tv_gi']
    combined_datasets = combine_datasets_by_stream_type(session_folders)

    averaged_data = {}

    non_feature_cols = ['stream_id', 'epoch', 'user_id', 'event_name', 'timestamp', 'event_timestamp', 'event_type', 'timedelta', 'metadata']

    for stream_type, dataset in combined_datasets.items():
        averaged_data[stream_type] = average_data_across_users(dataset, non_feature_cols)
        feature_columns = [col for col in averaged_data[stream_type].columns if
                           col not in non_feature_cols and averaged_data[stream_type][col].dtype != 'object']

        for feature in feature_columns:
            print(f"Plotting for {stream_type} data - Feature: {feature}")
            figures = plot_single_feature_per_session(averaged_data[stream_type], feature)
            titles = [f"{stream_type}_Session_{i + 1}_{feature}" for i in range(len(figures))]
            save_figures(figures, titles, f"{stream_type}_{feature}", "D:/Study Data/plots")


# def main():
#     session_folders = ['D:/Study Data/bz_fk', 'D:/Study Data/jk_ka', 'D:/Study Data/pb_kk', 'D:/Study Data/pd_pn', 'D:/Study Data/tv_gi']
#     combined_datasets = combine_datasets_by_stream_type(session_folders)
#
#     averaged_data = {}
#
#     non_feature_cols = ['stream_id', 'epoch', 'user_id', 'event_name', 'timestamp', 'event_timestamp', 'event_type',
#                         'timedelta', 'metadata']
#
#     for stream_type, dataset in combined_datasets.items():
#         # dataset.to_csv(f'D:/Study Data/{stream_type}_all_data.csv', index=False)
#         averaged_data[stream_type] = average_data_across_users(dataset, non_feature_cols)
#         print(averaged_data[stream_type].head())
#
#     selected_events = [
#         'start_BRUMS', 'parity_set_1', 'dualt_acc_1', 'individualization_acc_1',
#         'final_test_set_1_instr', 'final_test_instr', 'childs_pose_1',
#         'low_lunge_1', 'cobra_pose_1', 'low_lunge_2', 'cobra_pose_2',
#         'tree_pose_1', 'high_lunge_1', 'childs_pose_4', 'savasana_1', 'closing_chant_1'
#     ]
#
#     selected_types = ['Instruction Events', 'Sync Events', 'Zero Score Trials', 'Accuracy Events', 'Yoga Poses']

    # plot_features_per_session_and_phase(averaged_data['eeg'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['gsr'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['gyro'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['ppg'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['temp'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['acc'], non_feature_cols,
    #                             selected_events=selected_events)
    # plot_features_per_session_and_phase(averaged_data['bvp'], non_feature_cols,
    #                             selected_events=selected_events)

    # 'BVP', f'D:/Study Data/plots',
    # 'ACC', f'D:/Study Data/plots',
    # 'TEMP', f'D:/Study Data/plots',
    # 'PPG', f'D:/Study Data/plots',
    # 'GYRO', f'D:/Study Data/plots',
    # 'GSR', f'D:/Study Data/plots',
    # 'EEG', f'D:/Study Data/plots',

    # plot_individual_user_sensor_events(merged_acc, 'time_features_x_mean', 'kk', selected_events)



    # plot_combined_user_sensor_events_with_markers(merged_acc, 'time_features_x_mean', 'kk')


if __name__ == '__main__':
    main()



# def plot_features_for_all_users(merged_data, non_feature_columns,type, save_path , selected_events=None, selected_event_types=None):
#     """
#     Plots all features for each user in the merged data.
#
#     :param merged_data: DataFrame containing the merged data.
#     :param non_feature_columns: List of columns that are not features.
#     :param selected_events: (Optional) List of specific event names to include in the plot.
#     :param selected_event_types: (Optional) List of specific event types to include in the plot.
#     """
#     feature_columns = [col for col in merged_data.columns if col not in non_feature_columns]
#     users = merged_data['user_id'].unique()
#
#     for user_id in users:
#         for feature in feature_columns:
#             if selected_event_types:
#                 fig, title = plot_features_per_session_and_phase(merged_data, feature, user_id, selected_event_types)
#                 save_figures([fig], [title], f'{type}', save_path)
#             elif selected_events:
#                 fig, title = plot_features_per_session_and_phase(merged_data, feature, user_id, selected_events)
#                 save_figures([fig], [title], f'{type}', save_path)
#             else:
#                 # Default plot if no specific events or event types are provided
#                 fig, title = plot_features_per_session_and_phase(merged_data, feature, user_id, selected_events)
#                 save_figures([fig], [title], f'{type}', save_path)
#
#
# def plot_features_per_session_and_phase(averaged_data, non_feature_columns, selected_events=None):
#     """
#     Plots all features for each session and phase in the averaged data.
#
#     :param averaged_data: DataFrame containing the averaged data.
#     :param non_feature_columns: List of columns that are not features.
#     :param selected_events: (Optional) List of specific event names to include in the plot.
#     """
#     feature_columns = [col for col in averaged_data.columns if col not in non_feature_columns]
#     sessions = averaged_data['session'].unique()
#     phases = averaged_data['phase'].unique()
#
#     for session in sessions:
#         for phase in phases:
#             session_phase_data = averaged_data[
#                 (averaged_data['session'] == session) & (averaged_data['phase'] == phase)]
#             for feature in feature_columns:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sns.lineplot(data=session_phase_data, x='epoch', y=feature, ax=ax, marker='o', label=f'Session {session}, {phase}')
#
#                 # Marking the events if specified
#                 if selected_events:
#                     for _, row in session_phase_data.iterrows():
#                         if row['event_name'] in selected_events:
#                             ax.axvline(x=row['epoch'], color='grey', alpha=0.5, linestyle='--')
#                             ax.text(row['epoch'], ax.get_ylim()[1], row['event_name'], fontsize=9, verticalalignment='top', rotation=45)
#
#                 ax.set_title(f"{feature} Averages Over Sessions and Phases")
#                 ax.set_xlabel('Epoch')
#                 ax.set_ylabel(feature)
#                 ax.legend(loc='upper right')
#                 plt.show()
