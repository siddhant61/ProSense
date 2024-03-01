# ProSense (WIP)

## Introduction
ProSense is a powerful data analysis tool designed for processing and interpreting multimodal physiological data. It is capable of handling various data types such as EEG, PPG, ACC, GYRO, BVP, GSR, and TEMP, making it a versatile platform for researchers and practitioners in the field of psychophysiology.

## Features

- Data preprocessing techniques tailored for each physiological signal type.
- Advanced feature extraction algorithms to derive meaningful metrics from raw data.
- Automated and manual data visualization tools for thorough data inspection.
- Streamlined data analysis process from loading to transformation and processing.

## Getting Started
Clone the repository:
```
git clone https://github.com/siddhant61/StreamSense.git
```
Navigate to the StreamSense directory and install dependencies:
```
cd StreamSense
pip install -r requirements.txt
```
## Quickstart Guide
Running ProSense
To start the ProSense application, run:
```
python main.py
```

## Data Preprocessing and Feature Extraction

ProSense includes a variety of preprocessing and feature extraction techniques specific to each data type:

- **EEG Data**: Implements bandpass filtering, artifact removal, and spectral feature extraction.
- **PPG Data**: Focuses on heart rate variability and amplitude features.
- **ACC and GYRO Data**: Extracts both time and frequency domain features.
- **BVP Data**: Analyzes cardiovascular indicators such as HRV.
- **GSR Data**: Examines skin conductance responses.
- **TEMP Data**: Processes temperature-related data for variability and trends.
  ![Features](https://github.com/siddhant61/ProSense/blob/master/images/Slide31.JPG)

## Data Analysis Process Overview

ProSense follows a structured data analysis pipeline, which is outlined in the images provided. This pipeline includes:

- Data Loading
- Data Extraction
- Data Transformation
- Stream Specific Processing
- Automatic and Manual Visualization
- Outlier Rejection
- Identification of Significant Features
  ![System Architecture](https://github.com/siddhant61/ProSense/blob/master/images/Slide30.JPG)

## Visualization and Reporting

ProSense offers both automatic and manual visualization capabilities to facilitate the exploration and interpretation of the processed data.
Some of the plots inlcuding bar plots, time series plots, box plots from a study have been show below.
![Bar Plot](https://github.com/siddhant61/ProSense/blob/master/images/Slide35.JPG)
![Line Plot](https://github.com/siddhant61/ProSense/blob/master/images/Slide36.JPG)
![Line Plot Grouped](https://github.com/siddhant61/ProSense/blob/master/images/Slide41.JPG)
![Box Plot Phased](https://github.com/siddhant61/ProSense/blob/master/images/Slide37.JPG)
![Box Plot Events](https://github.com/siddhant61/ProSense/blob/master/images/Slide40.JPG)
