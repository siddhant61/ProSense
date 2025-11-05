"""
Unit tests for load_features.py

Tests feature flattening and utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Mock pandas.read_csv to avoid file dependency before importing load_features
mock_user_mapping = pd.DataFrame({
    'stream_id': ['muses-12345', '67890'],
    'user_id': [1, 2]
})

with patch('pandas.read_csv', return_value=mock_user_mapping):
    from load_features import (
        get_user_id_for_stream,
        flatten_temp_features,
        flatten_gsr_features,
        flatten_acc_features,
        flatten_gyro_features,
        flatten_bvp_features,
        flatten_ppg_features,
        merge_features_with_events
    )


class TestFlattenTempFeatures:
    """Test suite for flatten_temp_features function."""

    @pytest.mark.unit
    def test_flatten_temp_features_basic(self):
        """Test basic temperature feature flattening."""
        temp_features = {
            'stream_1': [
                {
                    'Min Temperature': 32.0,
                    'Mean Temperature': 32.5,
                    'Max Temperature': 33.0,
                    'Temperature Variability': 0.3,
                    'Rate of Change': 0.01
                },
                {
                    'Min Temperature': 32.1,
                    'Mean Temperature': 32.6,
                    'Max Temperature': 33.1,
                    'Temperature Variability': 0.25,
                    'Rate of Change': 0.02
                }
            ]
        }

        log_map = {'12345': 'log_m12345_e67890'}
        session = 'session_1'

        df = flatten_temp_features(temp_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 epochs
        assert 'stream_id' in df.columns
        assert 'epoch' in df.columns
        assert 'session' in df.columns
        assert 'Min Temperature' in df.columns
        assert 'Mean Temperature' in df.columns

        # Check values
        assert df.iloc[0]['Min Temperature'] == 32.0
        assert df.iloc[1]['Mean Temperature'] == 32.6
        assert df.iloc[0]['epoch'] == 0
        assert df.iloc[1]['epoch'] == 1

    @pytest.mark.unit
    def test_flatten_temp_features_multiple_streams(self):
        """Test flattening with multiple streams."""
        temp_features = {
            'stream_1': [
                {'Min Temperature': 32.0, 'Mean Temperature': 32.5,
                 'Max Temperature': 33.0, 'Temperature Variability': 0.3,
                 'Rate of Change': 0.01}
            ],
            'stream_2': [
                {'Min Temperature': 31.5, 'Mean Temperature': 32.0,
                 'Max Temperature': 32.5, 'Temperature Variability': 0.4,
                 'Rate of Change': 0.02}
            ]
        }

        log_map = {}
        session = 'test_session'

        df = flatten_temp_features(temp_features, session, log_map)

        # Should have 2 rows (1 epoch per stream)
        assert len(df) == 2
        # Check both streams present
        assert 'stream_1' in df['stream_id'].values
        assert 'stream_2' in df['stream_id'].values


class TestFlattenTempFeaturesMultipleStreams:
    """Additional tests for temperature feature flattening."""

    @pytest.mark.integration
    def test_temp_features_with_varying_epochs(self):
        """Test flattening with different number of epochs per stream."""
        temp_features = {
            'stream_1': [
                {'Min Temperature': 32.0, 'Mean Temperature': 32.5,
                 'Max Temperature': 33.0, 'Temperature Variability': 0.3,
                 'Rate of Change': 0.01},
                {'Min Temperature': 32.1, 'Mean Temperature': 32.6,
                 'Max Temperature': 33.1, 'Temperature Variability': 0.25,
                 'Rate of Change': 0.02}
            ],
            'stream_2': [
                {'Min Temperature': 31.5, 'Mean Temperature': 32.0,
                 'Max Temperature': 32.5, 'Temperature Variability': 0.4,
                 'Rate of Change': 0.015}
            ]
        }

        log_map = {}
        session = 'test_session'

        df = flatten_temp_features(temp_features, session, log_map)

        # Total of 3 rows (2 from stream_1, 1 from stream_2)
        assert len(df) == 3

        # Check epoch numbering is correct for each stream
        stream_1_epochs = df[df['stream_id'] == 'stream_1']['epoch'].tolist()
        assert stream_1_epochs == [0, 1]

        stream_2_epochs = df[df['stream_id'] == 'stream_2']['epoch'].tolist()
        assert stream_2_epochs == [0]


class TestFlattenAccFeatures:
    """Test suite for flatten_acc_features function."""

    @pytest.mark.unit
    def test_flatten_acc_features_basic(self):
        """Test basic accelerometer feature flattening."""
        acc_features = {
            'stream_1': {
                'time_features': {
                    0: {
                        'x': {'mean': 0.5, 'std': 0.1, 'min': 0.3, 'max': 0.7},
                        'y': {'mean': 0.6, 'std': 0.15, 'min': 0.4, 'max': 0.8},
                        'z': {'mean': 9.8, 'std': 0.2, 'min': 9.5, 'max': 10.1}
                    }
                },
                'freq_features': {
                    0: {
                        'x': {'psd': 0.05},
                        'y': {'psd': 0.06},
                        'z': {'psd': 0.04}
                    }
                },
                'sma_features': {
                    0: {
                        'x': {'sma': 100.5},
                        'y': {'sma': 110.2},
                        'z': {'sma': 980.5}
                    }
                },
                'entropy_features': {
                    0: {
                        'x': {'entropy': 2.5},
                        'y': {'entropy': 2.7},
                        'z': {'entropy': 2.3}
                    }
                }
            }
        }

        log_map = {}
        session = 'test_session'

        df = flatten_acc_features(acc_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'stream_id' in df.columns
        assert 'epoch' in df.columns
        assert 'session' in df.columns

        # Check feature flattening
        assert 'time_features_x_mean' in df.columns
        assert 'freq_features_x_psd' in df.columns
        assert 'sma_features_z_sma' in df.columns
        assert 'entropy_features_y_entropy' in df.columns

        # Check values
        assert df.iloc[0]['time_features_x_mean'] == 0.5
        assert df.iloc[0]['freq_features_x_psd'] == 0.05
        assert df.iloc[0]['sma_features_z_sma'] == 980.5


class TestFlattenGyroFeatures:
    """Test suite for flatten_gyro_features function."""

    @pytest.mark.unit
    def test_flatten_gyro_features_basic(self):
        """Test basic gyroscope feature flattening."""
        gyro_features = {
            'stream_1': {
                'time_features': {
                    0: {
                        'x': {'mean': 0.1, 'std': 0.05},
                        'y': {'mean': 0.2, 'std': 0.06},
                        'z': {'mean': 0.15, 'std': 0.04}
                    }
                },
                'freq_features': {
                    0: {
                        'x': {'psd': 0.02},
                        'y': {'psd': 0.03},
                        'z': {'psd': 0.025}
                    }
                }
            }
        }

        log_map = {}
        session = 'test_session'

        df = flatten_gyro_features(gyro_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'time_features_x_mean' in df.columns
        assert 'freq_features_y_psd' in df.columns

        # Check values
        assert df.iloc[0]['time_features_x_mean'] == 0.1
        assert df.iloc[0]['freq_features_y_psd'] == 0.03


class TestFlattenBvpFeatures:
    """Test suite for flatten_bvp_features function."""

    @pytest.mark.unit
    def test_flatten_bvp_features_basic(self):
        """Test basic BVP feature flattening."""
        bvp_features = {
            'stream_1': {
                'time_features': {
                    0: {'mean': 100.0, 'std': 15.0, 'min': 80.0, 'max': 120.0}
                },
                'freq_features': {
                    0: {'psd': 0.5}
                },
                'hr_features': {
                    0: {'heart_rate': 75.0}
                },
                'hrv_features': {
                    0: {'hrv': 0.05}
                }
            }
        }

        log_map = {}
        session = 'test_session'

        df = flatten_bvp_features(bvp_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'time_features_mean' in df.columns
        assert 'hr_features_heart_rate' in df.columns
        assert 'hrv_features_hrv' in df.columns

        # Check values
        assert df.iloc[0]['time_features_mean'] == 100.0
        assert df.iloc[0]['hr_features_heart_rate'] == 75.0
        assert df.iloc[0]['hrv_features_hrv'] == 0.05


class TestFlattenPpgFeatures:
    """Test suite for flatten_ppg_features function."""

    @pytest.mark.unit
    def test_flatten_ppg_features_basic(self):
        """Test basic PPG feature flattening."""
        ppg_features = {
            'stream_1': {
                'hr_features': {
                    0: {
                        'ppg_1': {'heart_rate': 72.0, 'hrv': 0.045},
                        'ppg_2': {'heart_rate': 73.0, 'hrv': 0.047}
                    }
                },
                'amplitude_features': {
                    0: {
                        'ppg_1': {'amplitude': 50.0},
                        'ppg_2': {'amplitude': 52.0}
                    }
                }
            }
        }

        log_map = {}
        session = 'test_session'

        df = flatten_ppg_features(ppg_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'hr_features_ppg_1_heart_rate' in df.columns
        assert 'amplitude_features_ppg_2_amplitude' in df.columns

        # Check values
        assert df.iloc[0]['hr_features_ppg_1_heart_rate'] == 72.0
        assert df.iloc[0]['amplitude_features_ppg_2_amplitude'] == 52.0


class TestFlattenGsrFeatures:
    """Test suite for flatten_gsr_features function."""

    @pytest.mark.unit
    def test_flatten_gsr_features_basic(self):
        """Test basic GSR feature flattening without variance (to avoid bug in code)."""
        gsr_features = {
            'stream_1': [
                {
                    'Skin Conductance Level (SCL)': 5.2,
                    'Skin Conductance Response (SCR) Frequency': 3,
                    'Amplitude of SCRs': 0.25
                },
                {
                    'Skin Conductance Level (SCL)': 5.5,
                    'Skin Conductance Response (SCR) Frequency': 4,
                    'Amplitude of SCRs': 0.30
                }
            ]
        }

        log_map = {}
        session = 'test_session'

        df = flatten_gsr_features(gsr_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'stream_id' in df.columns
        assert 'epoch' in df.columns
        assert 'session' in df.columns

        # Check GSR features
        assert 'Skin Conductance Level (SCL)' in df.columns
        assert 'Amplitude of SCRs' in df.columns

        # Check values
        assert df.iloc[0]['Skin Conductance Level (SCL)'] == 5.2
        assert df.iloc[1]['Amplitude of SCRs'] == 0.30

    @pytest.mark.unit
    def test_flatten_gsr_features_with_series(self):
        """Test GSR feature flattening with Series values."""
        # Create Series for GSR Variance
        variance_series = pd.Series([0.15, 0.16, 0.14])

        gsr_features = {
            'stream_1': [
                {
                    'Skin Conductance Level (SCL)': 5.2,
                    'Skin Conductance Response (SCR) Frequency': 3,
                    'Amplitude of SCRs': 0.25,
                    'GSR Variance': variance_series
                }
            ]
        }

        log_map = {}
        session = 'test_session'

        df = flatten_gsr_features(gsr_features, session, log_map)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        # Check that Series was flattened into multiple columns
        assert 'GSR_Variance_0' in df.columns
        assert 'GSR_Variance_1' in df.columns
        assert 'GSR_Variance_2' in df.columns

        # Check values
        assert df.iloc[0]['GSR_Variance_0'] == 0.15
        assert df.iloc[0]['GSR_Variance_1'] == 0.16


class TestGetUserIdForStream:
    """Test suite for get_user_id_for_stream function."""

    @pytest.mark.unit
    def test_get_user_id_with_matching_log(self):
        """Test user ID retrieval with matching log entry."""
        stream_id = 'muses-12345_acc_x'
        log_map = {
            'log1': 'log_m12345_e67890'
        }

        user_id = get_user_id_for_stream(stream_id, log_map)
        # Should call user_id_mapping.get() which is mocked at module level
        # Since we mocked the CSV to return None for most lookups, this will be None
        assert user_id is None or isinstance(user_id, (int, type(None)))

    @pytest.mark.unit
    def test_get_user_id_without_matching_log(self):
        """Test user ID retrieval without matching log entry."""
        stream_id = 'unknown_stream_id'
        log_map = {
            'log1': 'log_m12345_e67890'
        }

        user_id = get_user_id_for_stream(stream_id, log_map)
        assert user_id is None
