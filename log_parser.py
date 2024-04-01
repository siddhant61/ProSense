import ast
import csv
import json
import os
import pickle
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import logging

logger = logging.Logger()

# Predefined questionnaires
brums_questions = {
    "Q1": "Panicky",
    "Q2": "Lively",
    "Q3": "Confused",
    "Q4": "Worn Out",
    "Q5": "Depressed",
    "Q6": "Downhearted",
    "Q7": "Annoyed",
    "Q8": "Exhausted",
    "Q9": "Mixed-Up",
    "Q10": "Sleepy",
    "Q11": "Bitter",
    "Q12": "Unhappy",
    "Q13": "Anxious",
    "Q14": "Worried",
    "Q15": "Energetic",
    "Q16": "Miserable",
    "Q17": "Muddled",
    "Q18": "Nervous",
    "Q19": "Angry",
    "Q20": "Active",
    "Q21": "Tired",
    "Q22": "Bad Tempered",
    "Q23": "Alert",
    "Q24": "Uncertain"
}

sss_questions = {
    "Q1": "How sleepy do you feel right now?"
}

vasf_questions = {
    "Q1": "Feeling tired",
    "Q2": "Feeling sleepy",
    "Q3": "Feeling drowsy",
    "Q4": "Feeling fatigued",
    "Q5": "Feeling worn out",
    "Q6": "Feeling energetic",
    "Q7": "Feeling active",
    "Q8": "Feeling vigorous",
    "Q9": "Feeling efficient",
    "Q10": "Feeling lively",
    "Q11": "Feeling bushed",
    "Q12": "Feeling exhausted",
    "Q13": "Difficulty in keeping my eyes open",
    "Q14": "Difficulty in moving my body",
    "Q15": "Difficulty in concentrating",
    "Q16": "Difficulty in carrying on a conversation",
    "Q17": "Desire to close my eyes",
    "Q18": "Desire to lie down"
}

# For the Likert scale, we have 4 questions related to the participant's state.
likert_questions = {
    "Q1": "How sleepy do you feel?",
    "Q2": "How stressed do you feel?",
    "Q3": "How high is your mental workload?",
    "Q4": "How fatigued do you feel?"
}



def format_unix_timestamp(dt):
    return dt

def convert_to_dict(metadata):
    if isinstance(metadata, dict):
        return metadata
    if pd.isna(metadata) or metadata in ['{}', '']:
        return {}
    try:
        return ast.literal_eval(metadata)
    except (ValueError, SyntaxError):
        return {}


def custom_sort_key(item):
    # Extract timestamp and phase
    timestamp, details = item
    phase = details.get('phase', '')

    # Define a phase priority, ensuring 'ended' events are processed before 'started' events
    phase_priority = {'ended': 1, 'started': 0}

    # Using the phase_priority to adjust the timestamp slightly for sorting
    # We subtract a small value for 'ended' events so they come before 'started' events
    adjusted_timestamp = timestamp - (0.01 if phase in phase_priority else 0)

    return adjusted_timestamp, phase_priority.get(phase, 2)

class LogParser:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.logs = {}
        self.events = {}
        self.summaries = {}

    def parse_logs(self, log_lines):
        trial_data = {"parity_trials": {}, "nback_trials": {}, "dualt_trials": {},
                      "individualization_trials": {}, "final_test_trials": {}}
        forms = {}
        rest_blocks = {}
        accuracies = {}
        instructions = {}
        durations = {}
        sync_events = {}
        task_start = None
        task_end = None
        trial_attempts = {}

        current_trial_type = None
        current_trial_key = None

        for i, line in enumerate(log_lines):
            timestamp, content = line.split(",", 1)
            timestamp = datetime.fromtimestamp(float(timestamp)).timestamp()

            # Identify the start and end of a trial
            if "block_trial" in content:
                match = re.search(r"block_trial: (\w+)_trial_(\d+) (started|ended)", content)
                if match:
                    trial_type, trial_number, action = match.groups()
                    trial_key_base = f"{trial_type}_trial_{int(trial_number)+1}"

                    # Increment attempt number if a new trial is starting
                    if action == "started":
                        trial_attempts[trial_key_base] = trial_attempts.get(trial_key_base, 0) + 1

                    # Construct a unique key for the trial including its attempt number
                    trial_key = f"{trial_key_base}_set_{trial_attempts[trial_key_base]}"
                    current_trial_type = f"{trial_type}_trials"
                    current_trial_key = trial_key

                    if action == "started":
                        trial_info = {"start": timestamp, "response": None, "score": None, "end": None}
                        trial_data[current_trial_type][current_trial_key] = trial_info
                    elif action == "ended":
                        trial_data[current_trial_type][current_trial_key]["end"] = timestamp
                        # Reset current trial keys
                        current_trial_type = None
                        current_trial_key = None

            # Identify the start and end of a trial
            elif "block_sync" in content:
                sync_match = re.search(r"block_sync: (\w+)", content)
                if sync_match:
                    sync_type = sync_match.group(1)
                    if "started" in content:
                        sync_events[sync_type] = {"start": timestamp}
                        durations[sync_type] = {"start": timestamp}
                    elif "ended" in content:
                        sync_events[sync_type]["end"] = timestamp
                        durations[sync_type]["end"] = timestamp
                        durations[sync_type]["duration"] = durations[sync_type]["end"] - durations[sync_type]["start"]

            # Capture responses and scores
            elif "block_resp" in content or "block_score" in content:
                if "block_resp" in content:
                    response = content.split(":")[1].strip()
                    if current_trial_type and current_trial_key:
                        trial_data[current_trial_type][current_trial_key]["response"] = response
                if "block_score" in content:
                    score = float(content.split(":")[1].replace('%','').strip())
                    if current_trial_type and current_trial_key:
                        trial_data[current_trial_type][current_trial_key]["score"] = score

            # Handle accuracy and duration info
            elif "block_acc" in content:
                acc=0
                acc_match = re.search(r"block_acc: (\w+)", content)
                if acc_match:
                    acc_type = acc_match.group(1)
                    set = int(acc_type.split('_')[-1])+1
                    acc_type = acc_type.replace(acc_type.split('_')[-1], str(set))
                    if "started" in content:
                        accuracies[acc_type] = {"start": timestamp}
                    elif "ended" in content:
                        accuracies[acc_type]["end"] = timestamp
                        accuracies[acc_type]["accuracy"] = list(log_lines)[i-1].split(': ')[1].replace('%','')
                        acc = float(list(log_lines)[i-1].split(': ')[1].replace('%',''))
                    if acc >= 85 and "individualization" in content:
                        durations['stime_time'] = list(log_lines)[i-4].split('= ')[1].replace(' s','')
                        durations['resp_time'] = list(log_lines)[i-3].split('= ')[1].replace(' s', '')

            # Handle instrcution info
            elif "block_instr" in content:
                instr_match = re.search(r"block_instr: (\w+)", content)
                if instr_match:
                    instr_type = instr_match.group(1)
                    if len(instr_type.split('_'))>2 and instr_type.split('_')[-1] != 'SYNC':
                        set = int(instr_type.split('_')[-1]) + 1
                        instr_type = instr_type.replace(instr_type.split('_')[-1], str(set))
                    if "started" in content:
                        instructions[f'{instr_type}_instr'] = {"start": timestamp}
                        durations[f'{instr_type}_instr'] = {"start": timestamp}

                    elif "ended" in content:
                        instructions[f'{instr_type}_instr']["end"] = timestamp
                        durations[f'{instr_type}_instr']["end"] = timestamp
                        durations[f'{instr_type}_instr']["duration"] = durations[f'{instr_type}_instr']["end"] - \
                                                                     durations[f'{instr_type}_instr']["start"]
                        if "milestone" in content:
                            duration = list(log_lines)[i - 1].split(' ')[4]
                            durations[f'{instr_type}_instr']["duration"] = duration

            # Handle block info
            elif "block_info" in content and "completed" in content:
                info_match = re.search(r"block_info: completed ([\w_]+) trials", content)
                if info_match:
                    info_type = info_match.group(1)
                    duration_match = re.search(r"in ([\d.]+) minutes", content)
                    if duration_match:
                        duration = duration_match.group(1)
                        durations[info_type] = float(duration)

            # Handle form info
            elif "block_form" in content:
                form_match = re.search(r"block_form: (\w+)", content)
                if form_match:
                    form_type = form_match.group(1)
                    if len(form_type.split('_'))>2 and form_type.split('_')[-1] != 'VASF':
                        set = int(form_type.split('_')[-1]) + 1
                        form_type = form_type.replace(form_type.split('_')[-1], str(set))
                    if "started" in content:
                        start = i+1
                        forms[form_type] = {"start": timestamp}
                        durations[form_type] = {"start": timestamp}
                    elif "ended" in content:
                        end = i
                        forms[form_type]["end"] = timestamp
                        durations[form_type]["end"] = timestamp
                        durations[form_type]["duration"] = durations[form_type]["end"] - durations[form_type]["start"]
                        for j in range (start, end):
                            response = list(log_lines)[j].split(': ')[1]
                            question = list(log_lines)[j].split(': ')[0].split(',')[1]
                            forms[form_type][question] = response

            # Handle rest block info
            elif "block_rest" in content:
                block_match = re.search(r"block_rest: (\w+)", content)
                if block_match:
                    rest_type = block_match.group(1)
                    set = int(rest_type.split('_')[-1]) + 1
                    rest_type = rest_type.replace(rest_type.split('_')[-1], str(set))
                    if "started" in content:
                        rest_blocks[rest_type] = {"start": timestamp}
                        start = timestamp
                        durations[rest_type] = {"start": timestamp}
                    elif "ended" in content:
                        rest_blocks[rest_type]["end"] = timestamp
                        end = timestamp
                        durations[rest_type]["end"] = end
                        durations[rest_type]["duration"] = end-start

            # Handle block start info
            elif "block_start" in content:
                block_match = re.search(r"block_start: (\w+)", content)
                if block_match:
                    block_type = block_match.group(1)
                    if "block_start" and "started" in content:
                        durations[block_type] = {"start": timestamp}
                        task_start = timestamp
                    if "block_start" and "ended" in content:
                        durations[block_type]["end"] = timestamp
                        durations[block_type]["duration"] = durations[block_type]["end"] - durations[block_type]["start"]

            # Handle block end info
            elif "block_end" in content:
                block_match = re.search(r"block_end: (\w+)", content)
                if block_match:
                    block_type = block_match.group(1)
                    if "block_end" and "started" in content:
                        durations[block_type] = {"start": timestamp}
                    if "block_end" and "ended" in content:
                        durations[block_type]["end"] = timestamp
                        durations[block_type]["duration"] = durations[block_type]["end"]-durations[block_type]["start"]
                        task_end = timestamp
                        durations['cognitive_load_induction'] = {'start': task_start, 'end': task_end, 'duration': task_end-task_start}

        return trial_data, forms, sync_events,rest_blocks, accuracies, instructions, durations


    def structure_data(self, trial_data, forms, sync_events, rest_blocks, accuracies, instructions):
        structured_data = []

        for instruction, details in instructions.items():
            end_timestamp = details['end'] - 0.1
            structured_data.append((format_unix_timestamp(details['start']),
                                    {'marker': instruction, 'type': 'instruction', 'phase': 'started'}))
            data_without_start_end = {k: v for k, v in details.items() if k not in ['start', 'end']}
            structured_data.append((format_unix_timestamp(end_timestamp),
                                    {'marker': instruction, 'type': 'instruction', 'phase': 'ended'}))

        for trial_type, trials in trial_data.items():
            for trial, details in trials.items():
                # Create an entry for the start of the trial
                structured_data.append((format_unix_timestamp(details['start']), {
                    'marker': trial, 'type': trial_type, 'phase': 'started'
                }))
                end_timestamp = details['end'] - 0.1
                # Create an entry for the end of the trial, including additional data
                trial_data = {k: v for k, v in details.items() if k not in ['start', 'end']}
                structured_data.append((format_unix_timestamp(end_timestamp), {
                    'marker': trial, 'type': trial_type, 'phase': 'ended', 'data': trial_data
                }))

        for accuracy, details in accuracies.items():
            end_timestamp = details['end'] - 0.1
            structured_data.append(
                (format_unix_timestamp(details['start']), {'marker': accuracy, 'type': 'accuracy', 'phase': 'started'}))
            data_without_start_end = {k: v for k, v in details.items() if k not in ['start', 'end']}
            structured_data.append((format_unix_timestamp(end_timestamp),
                                    {'marker': accuracy, 'type': 'accuracy', 'phase': 'ended',
                                     'data': data_without_start_end}))

        for form, details in forms.items():
            end_timestamp = details['end'] - 0.1
            structured_data.append(
                (format_unix_timestamp(details['start']), {'marker': form, 'type': 'form', 'phase': 'started'}))

            # Filter out 'start' and 'end' keys from the details dictionary
            data = {k: v for k, v in details.items() if k not in ['start', 'end']}
            structured_data.append((format_unix_timestamp(end_timestamp),
                                    {'marker': form, 'type': 'form', 'phase': 'ended', 'data': data}))

        for rest_block, details in rest_blocks.items():
            end_timestamp = details['end'] - 0.1
            structured_data.append(
                (format_unix_timestamp(details['start']), {'marker': rest_block, 'type': 'rest_block', 'phase': 'started'}))
            data_without_start_end = {k: v for k, v in details.items() if k not in ['start', 'end']}
            structured_data.append((format_unix_timestamp(end_timestamp),
                                    {'marker': rest_block, 'type': 'rest_block', 'phase': 'ended',
                                     'data': data_without_start_end}))

        for sync_block, details in sync_events.items():
            end_timestamp = details['end'] - 0.1
            structured_data.append(
                (format_unix_timestamp(details['start']), {'marker': sync_block, 'type': 'sync_block', 'phase': 'started'}))
            data_without_start_end = {k: v for k, v in details.items() if k not in ['start', 'end']}
            structured_data.append((format_unix_timestamp(end_timestamp),
                                    {'marker': sync_block, 'type': 'sync_block', 'phase': 'ended'}))

        # Sort the list based on the custom sort key
        structured_data.sort(key=custom_sort_key)

        return structured_data

    def parse_durations(self, durations):
        structured_durations = []
        summary = []
        for duration, details in durations.items():
            if isinstance(details, dict) and 'start' in details and 'end' in details:
                structured_durations.append(
                    (format_unix_timestamp(details['start']),
                     {'marker': duration, 'type': 'duration', 'phase': 'started'})
                )
                end_timestamp = details['end'] - 0.1
                data_without_start_end = {k: v for k, v in details.items() if k not in ['start', 'end']}
                structured_durations.append(
                    (format_unix_timestamp(end_timestamp),
                     {'marker': duration, 'type': 'duration', 'phase': 'ended', 'duration': data_without_start_end['duration']})
                )
            else:
                summary.append({duration: details})

        # Sort the list based on the custom sort key
        structured_durations.sort(key=custom_sort_key)
        return summary, structured_durations

    def remove_duplicates(self, df):
        df['metadata'] = df['metadata'].apply(convert_to_dict)
        duplicates = df.duplicated(subset=['marker', 'start_time', 'end_time'], keep=False)

        # Define duplicate_rows here
        duplicate_rows = df[duplicates]

        def select_row_to_keep(group):
            # Keep the row with metadata if available
            with_metadata = group.dropna(subset=['metadata'])
            if not with_metadata.empty:
                # If both have metadata, prefer the one without 'duration'
                without_duration = with_metadata[~with_metadata['metadata'].apply(lambda x: 'duration' in x)]
                return without_duration.head(1) if not without_duration.empty else with_metadata.head(1)
            return group.head(1)

        # Apply the selection logic to each group of duplicates
        cleaned_duplicates = duplicate_rows.groupby(['marker', 'start_time', 'end_time']).apply(
            select_row_to_keep).reset_index(drop=True)
        df_cleaned = pd.concat([df.drop(duplicates[duplicates].index), cleaned_duplicates]).reset_index(drop=True)
        return df_cleaned.sort_values(by=['marker', 'start_time', 'end_time'])


    def merge_columns(self, column_1, column_2):
        if column_1:
            return column_1
        else:
            return column_2


    def extract_major_events_and_forms(self, data):
        events = {
            'Trial Sets': [],
            'Zero Score Trials': [],
            'Accuracy Events': [],
            'Sync Events': [],
            'Rest Events': [],
            'Instruction Events': [],
            'Form Responses': {
                'BRUMS': [],
                'SSS': [],
                'VASF': [],
                'LIKERT': []
            }
        }

        trial_set_start = None
        trial_set_type = None

        for index, row in data.iterrows():
            marker = row['marker']
            start_time = row['start_time']
            end_time = row['end_time']
            # Check the type of metadata and parse if it's a string
            if isinstance(row['metadata'], str):
                try:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                except json.JSONDecodeError:
                    # Handle cases where the metadata is not in JSON format
                    metadata = {'raw_metadata': row['metadata']}
            else:
                # Use metadata as is if it's already a dictionary
                metadata = row['metadata'] if row['metadata'] else {}

            # Calculate duration if metadata is empty
            if not metadata:
                metadata = {'duration': end_time - start_time}

            # Trial sets and final test trials
            if 'trial_1_set_' in marker or 'final_test_trial_1_set' in marker:
                trial_set_start = start_time
                trial_set_type = marker.split('_')[0]
                set = int(marker.split('_')[-1])
                if trial_set_type == 'nback':
                    trial_set_type = f'n-back_set_{set}'
                elif trial_set_type == 'Dualt':
                    trial_set_type = f'dual_task_set_{set}'
                else:
                    trial_set_type = f'{trial_set_type}_set_{set}'
            elif ('trial_20_set_' in marker or 'final_test_trial_60_set' in marker) and trial_set_start:
                events['Trial Sets'].append({'Type': trial_set_type, 'Start': trial_set_start, 'End': end_time})
                trial_set_start = None

            # Zero score trials
            if 'score' in metadata and metadata['score'] == 0:
                events['Zero Score Trials'].append({'Marker': marker,'Start': start_time, 'End': end_time})

            # Accuracy events
            if 'acc' in marker:
                events['Accuracy Events'].append(
                    {'Marker': marker, 'Start': start_time, 'End': end_time, 'Metadata': metadata})

            # Instruction events
            if 'instr' in marker.lower():
                if 'trial' in marker.lower():
                    marker = marker.replace('trial', 'set')
                events['Instruction Events'].append({'Marker': marker,'Start': start_time, 'End': end_time, 'Metadata': metadata})

            # Sync events
            if 'sync' in marker.lower() and 'instr' not in marker.lower():
                events['Sync Events'].append({'Marker': marker, 'Start': start_time, 'End': end_time, 'Metadata': metadata})

            # Rest Events
            if 'rest' in marker.lower() and 'instr' not in marker.lower():
                events['Rest Events'].append({'Marker': marker,'Start': start_time, 'End': end_time, 'Metadata': metadata})

            # Form responses
            if 'brums' in marker.lower() and 'instr' not in marker.lower():
                events['Form Responses']['BRUMS'].append(
                    {'Marker': marker, 'Start': start_time, 'End': end_time, 'Responses': metadata})
            elif 'sss' in marker.lower() and 'instr' not in marker.lower():
                events['Form Responses']['SSS'].append(
                    {'Marker': marker, 'Start': start_time, 'End': end_time, 'Responses': metadata})
            elif 'vasf' in marker.lower() and 'instr' not in marker.lower():
                events['Form Responses']['VASF'].append(
                    {'Marker': marker, 'Start': start_time, 'End': end_time, 'Responses': metadata})
            elif 'likert' in marker.lower() and 'instr' not in marker.lower():
                events['Form Responses']['LIKERT'].append(
                    {'Marker': marker, 'Start': start_time, 'End': end_time, 'Responses': metadata})


        return events

    def update_metadata(self, row):
        if not row['Metadata']:  # Check if Metadata is empty
            # Calculate duration and update Metadata
            row['Metadata'] = {'duration': row['End'] - row['Start']}
        return row

    def process_logs(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file_name)
                with open(file_path, newline='') as csvfile:
                    log_reader = csv.reader(csvfile, delimiter=',')
                    log_lines = [','.join(row).replace('Sstart_YNC', 'start_SYNC') for row in log_reader]



                trial_data, forms, sync_events, rest_blocks, accuracies, instructions, durations = self.parse_logs(
                    log_lines)

                # print("trials logs:\n", trial_data, '\n\nforms logs:\n', forms, '\n\nsync_events logs:\n',
                #       sync_events, '\n\nrest_blocks logs:\n', rest_blocks, '\n\naccuracies logs:\n',
                #       accuracies, '\n\ninstructions logs:\n', instructions, '\n\ndurations logs:\n',
                #       durations)    # Debug

                # Structured data
                data = self.structure_data(trial_data, forms, sync_events, rest_blocks, accuracies, instructions)
                self.logs[file_name.split('.')[0]] = {'data': {entry[0]: entry[1] for entry in data}}

                # Structured durations and summary
                summary, event_durations = self.parse_durations(durations)
                self.events[file_name.split('.')[0]] = {'data': {entry[0]: entry[1] for entry in event_durations}}
                self.summaries[file_name.split('.')[0]] = {'data': summary}

                if self.events:
                    pickle_file_path = self.folder_path + "ProcessedLogs/events_dataset.pkl"
                    try:
                        with open(pickle_file_path, 'wb') as f:
                            pickle.dump(self.events, f)
                            logger.info(f"Processed events and saved to dataset.")
                    except Exception as e:
                        logger.error(
                            f"Error saving dataset for events data: {e}. Traceback: {traceback.format_exc()}")

                if self.logs:
                    pickle_file_path = self.folder_path + "ProcessedLogs/logs_dataset.pkl"
                    try:
                        with open(pickle_file_path, 'wb') as f:
                            pickle.dump(self.logs, f)
                            logger.info(f"Processed logs and saved to dataset.")
                    except Exception as e:
                        logger.error(
                            f"Error saving dataset for logs data: {e}. Traceback: {traceback.format_exc()}")

                # structured data, events, and summaries
                structured_data = self.logs
                structured_durations = self.events
                # Dictionary to store combined events for each key
                combined_events_per_key = {}

                # Process each key separately
                for key in structured_durations.keys():
                    combined_events = []

                    # Process structured durations for this key
                    durations_data = structured_durations[key]['data']
                    events_df = pd.DataFrame.from_dict(durations_data, orient='index')
                    events_df = events_df.reset_index().rename(columns={'index': 'timestamp'})

                    for index, row in events_df.iterrows():
                        if row['phase'] == 'ended':
                            start_event = events_df[
                                (events_df['marker'] == row['marker']) & (events_df['phase'] == 'started')]
                            if not start_event.empty:
                                start_time = start_event.iloc[0]['timestamp']
                                end_time = row['timestamp']
                                duration = row['duration']
                                combined_events.append({
                                    'marker': row['marker'],
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'metadata': {'duration': duration}
                                })

                    # Process structured data for this key
                    logs_data = structured_data[key]['data']
                    logs_df = pd.DataFrame.from_dict(logs_data, orient='index')
                    logs_df = logs_df.reset_index().rename(columns={'index': 'timestamp'})

                    for marker, group in logs_df.groupby('marker'):
                        start_event = group[group['phase'] == 'started']
                        end_event = group[group['phase'] == 'ended']
                        if not start_event.empty and not end_event.empty:
                            start_time = start_event.iloc[0]['timestamp']
                            end_time = end_event.iloc[0]['timestamp']
                            metadata = end_event.iloc[0].get('data', {})
                            combined_events.append({
                                'marker': marker,
                                'start_time': start_time,
                                'end_time': end_time,
                                'metadata': metadata
                            })

                    # Convert to DataFrame and store in dictionary
                    combined_events_per_key[key] = pd.DataFrame(combined_events)
                    combined_events_per_key[key] = combined_events_per_key[key].sort_values(
                        by='start_time').reset_index(drop=True)

                    # Convert metadata from string to dictionary and remove duplicates
                    combined_events_per_key[key]['metadata'] = combined_events_per_key[key]['metadata'].apply(
                        convert_to_dict)
                    combined_events_per_key[key] = self.remove_duplicates(combined_events_per_key[key])

                    # Sort events by start_time
                    combined_events_per_key[key] = combined_events_per_key[key].sort_values(
                        by='start_time').reset_index(drop=True)

                # Save each DataFrame to CSV
                for key, df in combined_events_per_key.items():
                    events = self.extract_major_events_and_forms(df)
                    # Combine all events into a single DataFrame
                    major_events = []

                    # Process each category of events
                    for category in ['Trial Sets', 'Zero Score Trials', 'Instruction Events', 'Accuracy Events', 'Sync Events',
                                     'Rest Events']:
                        for event in events[category]:
                            event['Category'] = category
                            # Merge metadata and responses
                            event['Metadata'] = self.merge_columns(event.get('Metadata', {}),
                                                                             event.get('Responses', {}))
                            event['Marker'] = self.merge_columns(event.get('Marker'),
                                                                             event.get('Type'))
                            # Remove the Responses key as it's now merged into Metadata
                            event.pop('Responses', None)
                            event.pop('Type', None)
                            major_events.append(event)

                    # Process form responses
                    for form_type, responses in events['Form Responses'].items():
                        for response in responses:
                            response['Category'] = form_type
                            # Merge metadata and responses
                            response['Metadata'] = self.merge_columns(response.get('Metadata', {}),
                                                                                response.get('Responses', {}))
                            response.pop('Responses', None)
                            major_events.append(response)


                    # Create DataFrame
                    major_events_df = pd.DataFrame(major_events)

                    # Load yoga pose data
                    yoga_file_path = os.path.join('D:/Study Data/yoga_markers.csv')
                    yoga_df = pd.read_csv(yoga_file_path)

                    # Dictionary to keep track of pose iterations
                    pose_iterations = {}

                    # Calculate start time for the first yoga event
                    last_event_end_time = major_events_df['End'].max()
                    yoga_start_time = last_event_end_time + 5 * 60  # Adding 5 minutes

                    yoga_events = []

                    for index, row in yoga_df.iterrows():
                        pose_name = row['yoga pose'].lower().replace(' ', '_')
                        pose_duration = row['duration (seconds)']
                        pose_iteration = pose_iterations.get(pose_name, 0) + 1
                        pose_iterations[pose_name] = pose_iteration

                        # Calculate end time based on duration
                        yoga_end_time = yoga_start_time + pose_duration

                        # Create yoga pose event
                        yoga_event = {
                            'Marker': f"{pose_name}_{pose_iteration}",
                            'Category': 'Yoga Poses',
                            'Start': yoga_start_time,
                            'End': yoga_end_time,
                            'Metadata': {'duration': pose_duration}
                        }
                        yoga_events.append(yoga_event)

                        # Update start time for the next yoga pose
                        yoga_start_time = yoga_end_time

                    # Create DataFrame from yoga events
                    yoga_events_df = pd.DataFrame(yoga_events)

                    # Concatenate major_events_df with yoga_events_df
                    major_events_df = pd.concat([major_events_df, yoga_events_df], ignore_index=True)

                    major_events_df = major_events_df.apply(self.update_metadata, axis=1)

                    # Order by start time
                    major_events_df = major_events_df.sort_values(by='Start')

                    # Save to CSV
                    major_events_df.to_csv(f'{self.folder_path}/ProcessedLogs/{key}_events.csv', index=False)


folder_path = 'D:/Study Data/set_1/session_1/logs/'
output_folder = Path(f'{folder_path}ProcessedLogs/')
output_folder.mkdir(parents=True, exist_ok=True)
log_parser = LogParser(folder_path)
log_parser.process_logs()







