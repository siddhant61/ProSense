import json
import os
import pickle
from datetime import timedelta
from pathlib import Path
import pandas as pd
import mne
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from load_data import LoadData
from prepro_data import PreProData
from modalities.prepro_eeg import PreProEEG
from modalities.prepro_gsr import PreProGSR
from modalities.prepro_acc import PreProACC
from modalities.prepro_bvp import PreProBVP
from modalities.prepro_gyro import PreProGYRO
from modalities.prepro_ppg import PreProPPG
from modalities.prepro_temp import PreProTEMP
from modalities.featex_eeg import FeatExEEG
from modalities.featex_gsr import FeatExGSR
from modalities.featex_acc import FeatExACC
from modalities.featex_bvp import FeatExBVP
from modalities.featex_gyro import FeatExGYRO
from modalities.featex_ppg import FeatExPPG
from modalities.featex_temp import FeatExTEMP


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


##########################################EEG
def process_eeg(files, eeg_datasets, loader):

    # Step 1: Initialize PreProEEG with your dataset
    prepro = PreProEEG(eeg_datasets)
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "raw_data")
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/EEG")

    # Step 2: Apply Downsampling
    prepro.apply_downsampling(max_sfreq=200)
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "downsampled_data")
    save_figures(figs, titles, 'downsampled_data', f"{files}/Plots/EEG")

    # Step 3: Apply Notch Filter
    prepro.apply_notch_filter(50)
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "notch_filtered_data")
    save_figures(figs, titles, 'nf_data', f"{files}/Plots/EEG")

    # Step 4: Apply Bandpass Filter
    prepro.apply_bandpass_filter(1, 40)
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "bandpass_filtered_data")
    save_figures(figs, titles, 'bpf_data', f"{files}/Plots/EEG")

    # Step 5: Apply Artifact Removal
    prepro.apply_artifact_removal()
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "artifact_removed_data")
    save_figures(figs, titles, 'artrej_data', f"{files}/Plots/EEG")

    # Step 6: Apply Epoching
    prepro.apply_epoching(epoch_duration=5.0)
    figs, titles = prepro.plot_eeg_data(eeg_datasets, "epoched_data")
    save_figures(figs, titles, 'epoched_data', f"{files}/Plots/EEG")

    # # Step 8: Apply Baseline Correction
    # prepro.apply_baseline_correction()
    # print("BASELINE CORRECTED DATA")
    # prepro.plot_eeg_data(eeg_datasets)

    # # Step 7: Apply Artifact Rejection
    # prepro.apply_rejection(threshold=100e-6)
    # figs, titles = prepro.plot_eeg_data(eeg_datasets, "threshold_rejected_data")
    # save_figures(figs, titles, 'threshold_data', f"{files}/Plots")

    # Step 9: Check for Normality
    is_normal = prepro.check_normality(alpha=0.05)
    if is_normal:
        # Step 10: Apply Normalization
        prepro.apply_normalization()
        figs, titles = prepro.plot_eeg_data(eeg_datasets, "normalized_data")
        save_figures(figs, titles, 'norm_data', f"{files}/Plots/EEG")
    else:
        # Step 11: Apply Standardization
        prepro.apply_standardization()
        figs, titles = prepro.plot_eeg_data(eeg_datasets, "standardized_data")
        save_figures(figs, titles, 'norm_data', f"{files}/Plots/EEG")

    # Create an instance of the FeatEx class
    featex = FeatExEEG()

    # Extract features from the EEG data
    extracted_features = featex.extract_features(eeg_datasets)

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "eeg_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    total_times = []
    for file, data in extracted_features.items():
        total_times.append(data['total_epochs']*5)

    # Save all power band ratio plots
    ratio_names = ['alpha_theta_ratio', 'alpha_delta_ratio']
    figs, titles = featex.plot_all_power_band_ratios(extracted_features, ratio_names)
    save_figures(figs, titles, 'all_power_band_ratios', f"{files}/Plots/EEG")

    ratio_names = ['alpha_power']
    figs, titles = featex.plot_all_power_band_ratios(extracted_features, ratio_names)
    save_figures(figs, titles, 'alpha_power_band', f"{files}/Plots/EEG")

    ratio_names = ['delta_power']
    figs, titles = featex.plot_all_power_band_ratios(extracted_features, ratio_names)
    save_figures(figs, titles, 'delta_power_band', f"{files}/Plots/EEG")

    ratio_names = ['theta_power']
    figs, titles = featex.plot_all_power_band_ratios(extracted_features, ratio_names)
    save_figures(figs, titles, 'theta_power_band', f"{files}/Plots/EEG")

    ratio_names = ['low_beta_power']
    figs, titles = featex.plot_all_power_band_ratios(extracted_features, ratio_names)
    save_figures(figs, titles, 'low_beta_power_band', f"{files}/Plots/EEG")

    ratio_names = ['alpha_power', 'alpha_theta_ratio', 'alpha_delta_ratio']
    figs, titles = featex.plot_specific_epochs_power_band_ratios(extracted_features, [20,21,22,23, 24, 25, 26, 27, 28, 29] ,ratio_names)
    save_figures(figs, titles, 'power_bands_ratio_epoch', f"{files}/Plots/EEG")

    ratio_names = ['alpha_power']
    figs, titles = featex.plot_avg_power_band_ratios(extracted_features,ratio_names)
    save_figures(figs, titles, 'avg_power_band_ratios', f"{files}/Plots/EEG")

    # Save all spectral entropy plots
    figs, titles = featex.plot_all_spectral_entropies(extracted_features)
    save_figures(figs, titles, 'all_spectral_entropies', f"{files}/Plots/EEG")

    figs, titles = featex.plot_specific_epochs_spectral_entropy(extracted_features, [20,21,22,23, 24, 25, 26, 27, 28, 29] )
    save_figures(figs, titles, 'spectral_entropies_epoch', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_spectral_entropy(extracted_features)
    save_figures(figs, titles, 'avg_spectral_entropy', f"{files}/Plots/EEG")

    # Save all PSD value plots
    figs, titles = featex.plot_all_psd_values(extracted_features, ['mean_power', 'median_power', 'max_power'])
    save_figures(figs, titles, 'all_psd_values', f"{files}/Plots/EEG")

    # figs, titles = featex.plot_all_psd_values(extracted_features, ['max_power'])
    # save_figures(figs, titles, 'max_psd_values', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_psd_features(extracted_features, ['mean_power', 'median_power', 'max_power'])
    save_figures(figs, titles, 'avg_psd_features', f"{files}/Plots/EEG")

    # figs, titles = featex.plot_avg_psd_features(extracted_features, ['max_power'])
    # save_figures(figs, titles, 'avg_max_psd_features', f"{files}/Plots/EEG")

    # Save all coherence value plots
    figs, titles = featex.plot_all_coherence_values(extracted_features)
    save_figures(figs, titles, 'all_coherence_values', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_coherence_features(extracted_features)
    save_figures(figs, titles, 'avg_coherence_features', f"{files}/Plots/EEG")

    # Save all TFR value plots
    figs, titles = featex.plot_avg_tfr_heatmap(extracted_features)
    save_figures(figs, titles, 'avg_tfr_heatmaps', f"{files}/Plots/EEG")

    figs, titles = featex.plot_all_tfr_values(extracted_features)
    save_figures(figs, titles, 'all_tfr_values', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_tfr_values(extracted_features)
    save_figures(figs, titles, 'avg_tfr_values', f"{files}/Plots/EEG")

    # Save all statistical value plots
    figs, titles = featex.plot_all_stat_values(extracted_features, ['mean'])
    save_figures(figs, titles, 'all_stat_mean', f"{files}/Plots/EEG")

    figs, titles = featex.plot_all_stat_values(extracted_features, ['std'])
    save_figures(figs, titles, 'all_stat_std', f"{files}/Plots/EEG")

    figs, titles = featex.plot_all_stat_values(extracted_features, ['variance'])
    save_figures(figs, titles, 'all_stat_variance', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_statistical_features(extracted_features, ['mean', 'variance', 'std'])
    save_figures(figs, titles, 'avg_statistical_features', f"{files}/Plots/EEG")

    # Save all epoch stat plots
    # figs, titles = featex.plot_all_epoch_stats(extracted_features, ['epoch_mean', 'epoch_kurtosis', 'epoch_skewness'], ['AF7'])
    # save_figures(figs, titles, 'all_epoch_stats', f"{files}/Plots/EEG")
    #
    # figs, titles = featex.plot_all_epoch_stats(extracted_features, ['epoch_mean', 'epoch_kurtosis', 'epoch_skewness'], ['AF8'])
    # save_figures(figs, titles, 'all_epoch_stats', f"{files}/Plots/EEG")
    #
    # # Save all epoch stat plots
    # figs, titles = featex.plot_all_epoch_stats(extracted_features, ['epoch_mean', 'epoch_kurtosis', 'epoch_skewness'], ['TP9'])
    # save_figures(figs, titles, 'all_epoch_stats', f"{files}/Plots/EEG")

    # Save all epoch stat plots
    figs, titles = featex.plot_all_epoch_stats(extracted_features, ['epoch_mean', 'epoch_kurtosis', 'epoch_skewness'], ['AF7','AF8','TP9','TP10'])
    save_figures(figs, titles, 'all_epoch_stats', f"{files}/Plots/EEG")

    figs, titles = featex.plot_avg_epoch_features(extracted_features)
    save_figures(figs, titles, 'avg_epoch_features', f"{files}/Plots/EEG")



##########################################PPG
def process_ppg(files, ppg_datasets, loader):
    prepro = PreProPPG(ppg_datasets)
    figs, titles = prepro.plot_ppg_data(ppg_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/PPG")
    ppg_datasets = prepro.preprocess_ppg_data(5)
    figs, titles = prepro.plot_ppg_data(ppg_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/PPG")


    # Create an instance of the FeatEx class
    featex = FeatExPPG(ppg_datasets)

    # Extract features from the PPG data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "ppg_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    figs, titles = featex.plot_features(extracted_features)
    save_figures(figs, titles, 'features_over_time', f"{files}/Plots/PPG")

    figs, titles = featex.plot_features_over_time(extracted_features, 5)
    save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/PPG")



##########################################ACC
def process_acc(files, acc_datasets, loader):
    prepro = PreProACC(acc_datasets)
    figs, titles = prepro.plot_acc_data(acc_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/ACC")
    acc_datasets = prepro.preprocess_acc_data(5)
    figs, titles = prepro.plot_acc_data(acc_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/ACC")

    # Create an instance of the FeatEx class
    featex = FeatExACC(acc_datasets)

    # Extract features from the ACC data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "acc_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    figs, titles = featex.plot_features_over_epoch(extracted_features)

    save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/ACC")

    figs, titles = featex.plot_features_over_time(extracted_features, 5)
    save_figures(figs, titles, 'features_over_time', f"{files}/Plots/ACC")


###########################################GYRO
def process_gyro(files, gyro_datasets, loader):
    prepro = PreProGYRO(gyro_datasets)
    figs, titles = prepro.plot_gyro_data(gyro_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/GYRO")
    gyro_datasets = prepro.preprocess_gyro_data(5)
    figs, titles = prepro.plot_gyro_data(gyro_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/GYRO")


    # Create an instance of the FeatEx class
    featex = FeatExGYRO(gyro_datasets)

    # Extract features from the GYRO data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "gyro_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    plots = featex.plot_features(extracted_features)

    for stream, features in plots.items():
        for figs, titles in features:
            save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/GYRO")

    # figs, titles = featex.plot_features_over_time(extracted_features, 5)
    # save_figures(figs, titles, 'features_over_time', f"{files}/Plots/GYRO")



##########################################BVP
def process_bvp(files, bvp_datasets, loader):
    prepro = PreProBVP(bvp_datasets)
    figs, titles = prepro.plot_bvp_data(bvp_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/BVP")
    bvp_datasets = prepro.preprocess_bvp_data(5)
    figs, titles = prepro.plot_bvp_data(bvp_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/BVP")


    # Create an instance of the FeatEx class
    featex = FeatExBVP(bvp_datasets)

    # Extract features from the BVP data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "bvp_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    figs, titles = featex.plot_features_over_epoch(extracted_features)
    save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/BVP")

    figs, titles = featex.plot_features_over_time(extracted_features, 5)
    save_figures(figs, titles, 'features_over_time', f"{files}/Plots/BVP")



##########################################GSR
def process_gsr(files, gsr_datasets, loader):
    prepro = PreProGSR(gsr_datasets)
    figs, titles = prepro.plot_gsr_data(gsr_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/GSR")
    gsr_datasets = prepro.preprocess_gsr_data(5)
    figs, titles = prepro.plot_gsr_data(gsr_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/GSR")


    # Create an instance of the FeatEx class
    featex = FeatExGSR(gsr_datasets)

    # Extract features from the GSR data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "gsr_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    figs, titles = featex.plot_features_over_epoch(extracted_features)
    save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/GSR")

    figs, titles = featex.plot_features_over_time(extracted_features, 5)
    save_figures(figs, titles, 'features_over_time', f"{files}/Plots/GSR")




###########################################TEMP
def process_temp(files, temp_datasets, loader):
    prepro = PreProTEMP(temp_datasets)
    figs, titles = prepro.plot_temp_data(temp_datasets)
    save_figures(figs, titles, 'raw_data', f"{files}/Plots/TEMP")
    temp_datasets = prepro.preprocess_temp_data(5)
    figs, titles = prepro.plot_temp_data(temp_datasets)
    save_figures(figs, titles, 'preprocessed_data', f"{files}/Plots/TEMP")


    # Create an instance of the FeatEx class
    featex = FeatExTEMP(temp_datasets)

    # Extract features from the TEMP data
    extracted_features = featex.extract_features()

    output_folder = f"{files}/Features"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "temp_features.pkl"
    loader.save_pkl_dataset(extracted_features,output_file)

    figs, titles = featex.plot_features_over_epoch(extracted_features)
    save_figures(figs, titles, 'features_over_epochs', f"{files}/Plots/TEMP")

    figs, titles = featex.plot_features_over_time(extracted_features, 5)
    save_figures(figs, titles, 'features_over_time', f"{files}/Plots/TEMP")


if __name__ == '__main__':
    loader = LoadData()

    initial_path = "D:/Study Data/tv_gi/session_1/datasets/"
    inital_log_path = "D:/Study Data/tv_gi/session_1/logs/ProcessedLogs/"
    sessions = 5

    for i in range(5,sessions+1):

        folder_path = initial_path.replace('session_1', f'session_{i}')

        log_path = inital_log_path.replace('session_1', f'session_{i}')

        files = loader.process_datasets(folder_path)

        data_loader = PreProData(files)

        events, datasets, log_map = data_loader.load_datasets(log_path)

        eeg_datasets = {}
        ppg_datasets = {}
        acc_datasets = {}
        gyro_datasets = {}
        bvp_datasets = {}
        temp_datasets = {}
        gsr_datasets = {}

        print(datasets.keys())
        output_folder = f"{files}/Events"
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / "events.pkl"
        loader.save_pkl_dataset(events,output_file)
        loader.save_pkl_dataset(log_map, output_folder / "log_map.pkl")

        for stream_id,data in datasets.items():
            if 'EEG' in stream_id:
                eeg_datasets[stream_id] = data
            elif 'PPG' in stream_id:
                ppg_datasets[stream_id] = data
            elif 'ACC' in stream_id:
                acc_datasets[stream_id] = data
            elif 'GYRO' in stream_id:
                gyro_datasets[stream_id] = data
            elif 'BVP' in stream_id:
                bvp_datasets[stream_id] = data
            elif 'TEMP' in stream_id:
                temp_datasets[stream_id] = data
            elif 'GSR' in stream_id:
                gsr_datasets[stream_id] = data

        process_eeg(files, eeg_datasets, loader)

        process_ppg(files, ppg_datasets, loader)

        process_acc(files, acc_datasets, loader)

        process_gyro(files, gyro_datasets, loader)

        process_bvp(files, bvp_datasets, loader)

        process_gsr(files, gsr_datasets, loader)

        process_temp(files, temp_datasets, loader)

