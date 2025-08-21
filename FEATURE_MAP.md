# Feature-to-Code Mapping

This document maps the intended features of the ProSense project to the specific source code files and classes that implement them.

## Data Loading and Orchestration

| Feature | File | Class/Function |
| :--- | :--- | :--- |
| Main application logic | `main.py` | `if __name__ == '__main__':` |
| Data loading from files | `load_data.py` | `LoadData` |
| Initial data preparation | `prepro_data.py` | `PreProData` |

## Modality-Specific Processing

### EEG (Electroencephalography)

| Feature | File | Class | Method |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | `modalities/prepro_eeg.py` | `PreProEEG` | |
| Bandpass Filtering | | | `apply_bandpass_filter` |
| Downsampling | | | `apply_downsampling` |
| ICA Artifact Removal | | | `apply_artifact_removal` |
| Epoching/Segmentation | | | `apply_epoching` |
| Normalization | | | `apply_normalization` |
| **Feature Extraction** | `modalities/featex_eeg.py` | `FeatExEEG` | |
| Power Band Ratios | | | `extract_power_band_ratios` |
| Spectral Entropy | | | `extract_spectral_entropy` |
| Time-Frequency Features | | | `extract_time_frequency_features` |

### PPG (Photoplethysmography)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_ppg.py` | `PreProPPG` |
| Feature Extraction | `modalities/featex_ppg.py` | `FeatExPPG` |

### ACC (Accelerometer)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_acc.py` | `PreProACC` |
| Feature Extraction | `modalities/featex_acc.py` | `FeatExACC` |

### GYRO (Gyroscope)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_gyro.py` | `PreProGYRO` |
| Feature Extraction | `modalities/featex_gyro.py` | `FeatExGYRO` |

### BVP (Blood Volume Pulse)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_bvp.py` | `PreProBVP` |
| Feature Extraction | `modalities/featex_bvp.py` | `FeatExBVP` |

### GSR (Galvanic Skin Response)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_gsr.py` | `PreProGSR` |
| Feature Extraction | `modalities/featex_gsr.py` | `FeatExGSR` |

### TEMP (Temperature)

| Feature | File | Class |
| :--- | :--- | :--- |
| Preprocessing | `modalities/prepro_temp.py` | `PreProTEMP` |
| Feature Extraction | `modalities/featex_temp.py` | `FeatExTEMP` |
