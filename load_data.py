import json
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
import mne
import math
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


class LoadData:
    def __init__(self):
        self.channel_mapping = {
            'ACC': ['X', 'Y', 'Z'],
            'GYRO': ['X', 'Y', 'Z'],
            'EEG': ['AF7', 'AF8', 'TP9', 'TP10', 'R_AUX'],
            'TAG': ['TAG'],
            'GSR': ['GSR'],
            'BVP': ['BVP'],
            'TEMP': ['TEMP'],
            'PPG': ['Ambient', 'IR', 'Red']
        }

    # ===== INPUT VALIDATION METHODS =====

    def validate_file_exists(self, file_path):
        """
        Validate that a file exists at the given path.

        Args:
            file_path: Path to the file to check

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the path is empty or invalid
        """
        if not file_path:
            raise ValueError("File path cannot be empty")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return True

    def validate_directory_exists(self, dir_path):
        """
        Validate that a directory exists at the given path.

        Args:
            dir_path: Path to the directory to check

        Raises:
            FileNotFoundError: If the directory does not exist
            ValueError: If the path is empty or invalid
        """
        if not dir_path:
            raise ValueError("Directory path cannot be empty")

        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        return True

    def validate_file_extension(self, file_path, expected_ext):
        """
        Validate that a file has the expected extension.

        Args:
            file_path: Path to the file
            expected_ext: Expected extension (e.g., '.pkl', '.csv')

        Raises:
            ValueError: If the file extension doesn't match
        """
        path = Path(file_path)
        if path.suffix.lower() != expected_ext.lower():
            raise ValueError(
                f"Invalid file extension. Expected '{expected_ext}', got '{path.suffix}'"
            )
        return True

    def validate_dataframe(self, data, min_rows=1, required_columns=None):
        """
        Validate that data is a proper DataFrame with expected structure.

        Args:
            data: Data to validate
            min_rows: Minimum number of rows required
            required_columns: List of required column names

        Raises:
            TypeError: If data is not a DataFrame
            ValueError: If DataFrame doesn't meet requirements
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

        if len(data) < min_rows:
            raise ValueError(
                f"DataFrame has insufficient data: {len(data)} rows (minimum: {min_rows})"
            )

        if data.empty:
            raise ValueError("DataFrame is empty")

        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(
                    f"DataFrame missing required columns: {missing_cols}"
                )

        return True

    def validate_timestamps(self, timestamps, check_monotonic=True):
        """
        Validate that timestamps are valid and properly formatted.

        Args:
            timestamps: Array or Series of timestamps
            check_monotonic: Whether to check if timestamps are monotonically increasing

        Raises:
            ValueError: If timestamps are invalid
        """
        if len(timestamps) == 0:
            raise ValueError("Timestamp array is empty")

        # Check for NaN or None values
        if pd.isna(timestamps).any():
            raise ValueError("Timestamps contain NaN or None values")

        # Check if timestamps are numeric
        try:
            timestamps_numeric = pd.to_numeric(timestamps, errors='coerce')
            if timestamps_numeric.isna().any():
                raise ValueError("Timestamps contain non-numeric values")
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")

        # Check if timestamps are monotonically increasing
        if check_monotonic:
            if not np.all(np.diff(timestamps) > 0):
                raise ValueError("Timestamps are not monotonically increasing")

        return True

    def validate_sampling_rate(self, sfreq, min_sfreq=1, max_sfreq=10000):
        """
        Validate that sampling rate is within reasonable bounds.

        Args:
            sfreq: Sampling frequency to validate
            min_sfreq: Minimum acceptable sampling rate (Hz)
            max_sfreq: Maximum acceptable sampling rate (Hz)

        Raises:
            ValueError: If sampling rate is invalid
        """
        if sfreq is None:
            raise ValueError("Sampling rate cannot be None")

        try:
            sfreq = float(sfreq)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid sampling rate type: {type(sfreq).__name__}")

        if not (min_sfreq <= sfreq <= max_sfreq):
            raise ValueError(
                f"Sampling rate {sfreq} Hz is outside valid range "
                f"[{min_sfreq}, {max_sfreq}] Hz"
            )

        if sfreq <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sfreq}")

        return True

    def validate_data_array(self, data, min_samples=1, max_channels=None):
        """
        Validate that data array has valid shape and values.

        Args:
            data: Numpy array or similar
            min_samples: Minimum number of samples required
            max_channels: Maximum number of channels allowed

        Raises:
            ValueError: If data array is invalid
        """
        if data is None:
            raise ValueError("Data array cannot be None")

        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(
                f"Expected numpy array or pandas object, got {type(data).__name__}"
            )

        if isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError("Data array is empty")

            if data.ndim > 0 and data.shape[-1] < min_samples:
                raise ValueError(
                    f"Insufficient samples: {data.shape[-1]} (minimum: {min_samples})"
                )

            if max_channels and data.ndim > 1 and data.shape[0] > max_channels:
                raise ValueError(
                    f"Too many channels: {data.shape[0]} (maximum: {max_channels})"
                )

            # Check for NaN or Inf values
            if np.any(np.isnan(data)):
                raise ValueError("Data contains NaN values")

            if np.any(np.isinf(data)):
                raise ValueError("Data contains Inf values")

        return True

    def load_csv(self, file_path, columns):
        """Load CSV file with validation."""
        # Validate input
        self.validate_file_exists(file_path)
        self.validate_file_extension(file_path, '.csv')

        # Load data
        data = pd.read_csv(file_path, usecols=columns)
        data = data.dropna()  # Remove rows with NA values

        # Validate loaded data
        self.validate_dataframe(data, min_rows=1, required_columns=columns)

        return data

    def calculate_sfreq(self, timestamps):
        """Calculate sampling frequency from timestamps with validation."""
        try:
            # Validate timestamps
            self.validate_timestamps(timestamps, check_monotonic=True)

            # Convert float Unix timestamps to datetime objects
            timestamps_datetime = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

            # Calculate the differences between consecutive timestamps
            time_diff = np.diff(timestamps_datetime)
            # Convert datetime.timedelta to seconds
            time_diff = [td.total_seconds() for td in time_diff]
            avg_time_diff = np.mean(time_diff)

            if avg_time_diff == 0:
                raise ValueError("Invalid timestamps: Cannot calculate sampling frequency.")

            sfreq = int(1 / avg_time_diff)

            # Validate calculated sampling rate
            self.validate_sampling_rate(sfreq)

            return sfreq
        except ValueError as e:
            print(f"Validation or timestamp conversion error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in calculate_sfreq: {e}")
            raise

    def visualize_eeg_data(self, dataset):
        fig, axes = plt.subplots(len(dataset), figsize=(12, 6 * len(dataset)), sharex=True)
        fig.subplots_adjust(hspace=8)  # Adjust the spacing between subplots
        fig.tight_layout(pad=8.0)

        for i, (filename, data) in enumerate(dataset.items()):
            eeg_data = data['data'].get_data()
            channels = data['data'].ch_names[1:]  # Exclude the "TimeStamp" channel
            num_channels = len(channels)

            for j, channel in enumerate(channels):
                ax = axes[i] if len(dataset) > 1 else axes[j]
                ax.hist(eeg_data[j], bins=50, alpha=0.7, label=channel)

                # Add mean, std, skew, and kurtosis text to the plot
                mean = np.mean(eeg_data[j])
                std = np.std(eeg_data[j])
                skewness = skew(eeg_data[j])  # Use skew from scipy.stats
                kurt = kurtosis(eeg_data[j])
                ax.text(0.6, 0.8, f"Mean: {mean:.2f}\nStd: {std:.2f}\nSkewness: {skewness:.2f}\nKurtosis: {kurt:.2f}",
                        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

            ax.set_title(f"EEG Data - File: {filename} (sfreq: {data['sfreq']})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def convert_to_mne(self, data, sfreq):
        # Convert data to MNE format
        # Data shape is (n_channels, n_samples)
        raw = mne.io.RawArray(data, info=mne.create_info(ch_names=["AF7", "AF8", "TP9", "TP10"],
                                                         sfreq=sfreq, ch_types='eeg'))
        return raw

    def format_dataset(self, file_path):

        dataset = {}

        # Load the dataset from pickle file
        data = self.load_pkl_dataset(file_path)

        # Ensure the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Loaded dataset is not a pandas DataFrame.")

        # Check if columns are unnamed and name them accordingly
        if isinstance(data.columns, pd.RangeIndex):
            expected_channels = ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10', 'R_AUX']
            if len(data.columns) != len(expected_channels):
                raise ValueError(f"Data has {len(data.columns)} columns, but {len(expected_channels)} were expected.")
            data.columns = expected_channels

        # Drop the R_AUX channel if it's present
        if "R_AUX" in data.columns:
            data = data.drop(columns=["R_AUX"])

        # Expected channel names after removing R_AUX
        ch_names = ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10']

        # Ensure data has the expected channels
        if set(ch_names) != set(data.columns):
            raise ValueError(f"Data columns {data.columns} don't match expected channels {ch_names}.")

        # Calculate the sampling frequency from the data's index (timestamps)
        sfreq = self.calculate_sfreq(data.index)

        # Ensure sfreq is not None before proceeding
        if sfreq is None:
            raise ValueError("Couldn't calculate the sampling frequency.")

        # Convert data to MNE format
        mne_data = self.convert_to_mne(data.values.T, sfreq)

        # Extract the name of the file without its extension for dataset key
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        dataset[dataset_name] = {'data': mne_data, 'sfreq': sfreq}

        return dataset

    def save_pkl_dataset(self, all_features, file_path):
        """
        Save the all_features dictionary to a file using pickle.

        WARNING: Pickle is not secure against malicious data. Consider using
        safer formats (HDF5, Parquet) for production use.

        Args:
            all_features: Data to save
            file_path: Path where to save the file

        Raises:
            ValueError: If all_features is None or file_path is invalid
        """
        if all_features is None:
            raise ValueError("Cannot save None data")

        if not file_path:
            raise ValueError("File path cannot be empty")

        # Ensure parent directory exists
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(all_features, f)

    def load_pkl_dataset(self, file_path):
        """
        Load the all_features dictionary from a file using pickle.

        WARNING: Pickle is not secure. Only load data from trusted sources.
        Consider migrating to safer formats (HDF5, Parquet).

        Args:
            file_path: Path to the pickle file

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is wrong or data is invalid
        """
        # Validate input
        self.validate_file_exists(file_path)
        self.validate_file_extension(file_path, '.pkl')

        # Load data
        try:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Failed to unpickle file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading pickle file: {e}")

        # Validate loaded data
        if dataset is None:
            raise ValueError("Loaded dataset is None")

        return dataset

    def save_fif_dataset(self, dataset, filepath):
        for file, data in dataset.items():
            raw = data['data']
            sfreq = data['sfreq']
            raw.save(f'{filepath}/{file}_raw.fif', overwrite=True)
            if "epochs" in data.keys():
                epochs = data['epochs']
                epochs.save(f'{filepath}/{file}_epo.fif', overwrite=True)
            with open(f'{filepath}/{file}_sfreq.txt', 'w', encoding='utf-8') as f:
                f.write(str(sfreq))

    def load_fif_dataset(self, filepath):
        dataset = {}
        fif_files = []
        epo_files = []
        txt_files = []
        files = []
        for file in os.listdir(filepath):
            if ".fif" in file:
                if "raw" in file:
                    fif_files.append(mne.io.read_raw_fif(f'{filepath}/{file}'))
                    s = file
                    c = '_'
                    n = [pos for pos, char in enumerate(s) if char == c][1]
                    file = file[0:n]
                    files.append(file)
                else:
                    epo_files.append(mne.read_epochs(f'{filepath}/{file}'))
            else:
                with open(f'{filepath}/{file}', 'r', encoding='utf-8') as f:
                    txt_files.append(float(f.read()))

        if len(epo_files) != 0:
            for i in range(len(files)):
                dataset[files[i]] = {'data': fif_files[i], 'epochs': epo_files[i], 'sfreq': txt_files[i]}
        else:
            for i in range(len(files)):
                dataset[files[i]] = {'data': fif_files[i], 'sfreq': txt_files[i]}

        return dataset

    def process_datasets(self, folder_path):
        """
        Process datasets from a folder with validation.

        Args:
            folder_path: Path to folder containing datasets

        Returns:
            Path to output folder

        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If folder is invalid
        """
        # Validate input
        self.validate_directory_exists(folder_path)

        output_folder = Path(folder_path, "ProcessedData")
        output_folder.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(folder_path):
            if file.endswith('.pkl'):
                dataset_type = os.path.splitext(file)[0]
                dataset_path = os.path.join(folder_path, file)

                # Use validated load method
                dataset = self.load_pkl_dataset(dataset_path)

                for dataset_name, content in dataset.items():
                    data_file_path = output_folder / f"{dataset_name}.csv"
                    meta_file_path = output_folder / f"{dataset_name}_metadata.json"

                    eeg_df = content['data']  # Assuming it's already a DataFrame
                    gaps = content.get('gaps', [])

                    eeg_df = eeg_df.reset_index().rename(columns={'index': 'timestamp'})
                    eeg_df.to_csv(data_file_path, index=False)

                    metadata = {"sfreq": content.get('sfreq', None), "gaps": gaps.tolist()}
                    with open(meta_file_path, 'w') as meta_file:
                        json.dump(metadata, meta_file)

        return output_folder

    def load_files(self, path):
        data_dir = path
        file_names = os.listdir(data_dir)
        dataset = {}

        # General function to compute timestamps
        def compute_timestamps(length, sfreq):
            return np.arange(0, length) / sfreq

        for file in file_names:
            # Load data without specifying columns to infer data type from shape or content
            df = pd.read_csv(os.path.join(data_dir, file), header=None)

            # Check if the data has headers by seeing if the first entry is a timestamp (numeric)
            if not pd.to_numeric(df.iloc[0, 0], errors='coerce'):
                headers = df.iloc[0]
                df = df[1:]
                df.columns = headers
            else:
                # If no headers, infer data type based on shape
                if df.shape[1] == 3:  # Assuming ACC or GYRO based on 3 columns
                    # Use variance to distinguish between ACC and GYRO
                    # Accelerometer data typically has lower variance than gyroscope data
                    if df.var().mean() < 1:  # Threshold can be adjusted based on real data insights
                        df.columns = ["ACC_X", "ACC_Y", "ACC_Z"]
                    else:
                        df.columns = ["GYR_X", "GYR_Y", "GYR_Z"]
                elif df.shape[1] == 1:  # BVP, GSR, or TEMP
                    # Use range to distinguish between BVP, GSR, and TEMP
                    data_range = df.max() - df.min()
                    if data_range.values[0] < 10:  # Threshold for TEMP
                        df.columns = ["TEMP"]
                    elif data_range.values[0] < 100:  # Threshold for GSR
                        df.columns = ["GSR"]
                    else:
                        df.columns = ["BVP"]

                # Calculate timestamps
                sfreq = self.calculate_sfreq(compute_timestamps(len(df), sfreq=256))  # Assuming a default of 256Hz
                timestamps = compute_timestamps(len(df), sfreq)
                df.insert(0, "TimeStamp", timestamps)

            sfreq = self.calculate_sfreq(df['TimeStamp'])
            mne_data = self.convert_to_mne(df.drop('TimeStamp', axis=1).values.T, sfreq)

            # Extract the name of the file without its extension for dataset key
            dataset_name = os.path.splitext(file)[0]
            dataset[dataset_name] = {'data': mne_data, 'sfreq': sfreq}

        #self.visualize_eeg_data(dataset)
        return dataset


    def detect_spikes(self, data, threshold=2.5):
        """ Detect significant spikes in the accelerometer and gyroscope data. """
        magnitude = np.sqrt(np.sum(data**2, axis=1))
        spike_indices = np.where(magnitude > threshold)[0]
        start_time = spike_indices[0]
        end_time = spike_indices[-1]
        return start_time, end_time

    def slice_signals(self, dataset, start_time, end_time):
        """ Slice all signals in the dataset based on the provided start and end times. """
        sliced_dataset = {}
        for key, data in dataset.items():
            sliced_data = data[start_time:end_time]
            sliced_dataset[key] = sliced_data
        return sliced_dataset