"""
Tests for correlate_features.py

This module tests feature correlation, encoding, grouping, and visualization functions.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from correlate_features import (
    merge_features_with_events,
    calculate_correlation,
    visualize_correlation,
    create_binary_indicators,
    encode_events,
    get_user_id_for_stream,
    group_datasets_by_user_id,
    calculate_cross_stream_correlation,
    plot_individual_user_sensor_events,
    plot_combined_user_sensor_events_with_markers,
    plot_features_for_all_users,
    save_figures
)


class TestMergeFeaturesWithEvents:
    """Test suite for merge_features_with_events function."""

    @pytest.mark.unit
    def test_merge_basic(self):
        """Test basic merge operation."""
        feature_df = pd.DataFrame({
            'stream_id': ['s1', 's2'],
            'epoch': [0, 1],
            'feature1': [10.0, 20.0]
        })
        event_df = pd.DataFrame({
            'stream_id': ['s1', 's2'],
            'epoch': [0, 1],
            'event_name': ['event1', 'event2']
        })

        result = merge_features_with_events(feature_df, event_df)

        assert len(result) == 2
        assert 'feature1' in result.columns
        assert 'event_name' in result.columns
        assert result.iloc[0]['feature1'] == 10.0
        assert result.iloc[0]['event_name'] == 'event1'

    @pytest.mark.unit
    def test_merge_partial_match(self):
        """Test merge when only some rows match."""
        feature_df = pd.DataFrame({
            'stream_id': ['s1', 's2', 's3'],
            'epoch': [0, 1, 2],
            'feature1': [10.0, 20.0, 30.0]
        })
        event_df = pd.DataFrame({
            'stream_id': ['s1', 's2'],
            'epoch': [0, 1],
            'event_name': ['event1', 'event2']
        })

        result = merge_features_with_events(feature_df, event_df)

        assert len(result) == 2  # Only matching rows
        assert 's3' not in result['stream_id'].values

    @pytest.mark.unit
    def test_merge_empty_dataframes(self):
        """Test merge with empty DataFrames."""
        feature_df = pd.DataFrame(columns=['stream_id', 'epoch', 'feature1'])
        event_df = pd.DataFrame(columns=['stream_id', 'epoch', 'event_name'])

        result = merge_features_with_events(feature_df, event_df)

        assert len(result) == 0
        assert 'stream_id' in result.columns
        assert 'epoch' in result.columns


class TestCalculateCorrelation:
    """Test suite for calculate_correlation function."""

    @pytest.mark.unit
    def test_calculate_correlation_basic(self):
        """Test basic correlation calculation."""
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        event_indicators_df = pd.DataFrame({
            'event_A': [1, 1, 0, 0, 0],
            'event_B': [0, 0, 1, 1, 1]
        })

        result = calculate_correlation(features_df, event_indicators_df, threshold=0.3)

        assert isinstance(result, pd.DataFrame)
        # Features should correlate with at least one event

    @pytest.mark.unit
    def test_calculate_correlation_with_threshold(self):
        """Test correlation calculation with high threshold."""
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 1, 1, 1, 1]  # Low variance
        })
        event_indicators_df = pd.DataFrame({
            'event_A': [1, 0, 1, 0, 1]
        })

        result = calculate_correlation(features_df, event_indicators_df, threshold=0.9)

        # High threshold should filter out weak correlations
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_calculate_correlation_no_numeric_features(self):
        """Test correlation with non-numeric features."""
        features_df = pd.DataFrame({
            'feature1': ['a', 'b', 'c', 'd', 'e']
        })
        event_indicators_df = pd.DataFrame({
            'event_A': [1, 0, 1, 0, 1]
        })

        result = calculate_correlation(features_df, event_indicators_df, threshold=0.3)

        assert result.empty  # No numeric columns to correlate


class TestVisualizeCorrelation:
    """Test suite for visualize_correlation function."""

    @pytest.mark.unit
    def test_visualize_correlation_basic(self, monkeypatch):
        """Test basic correlation visualization."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        correlation_df = pd.DataFrame({
            'event_A': [0.8, 0.5],
            'event_B': [0.3, 0.9]
        }, index=['feature1', 'feature2'])

        visualize_correlation(correlation_df, title="Test Correlation")

        assert True  # Execution without error
        plt.close('all')

    @pytest.mark.unit
    def test_visualize_correlation_empty_dataframe(self, capsys):
        """Test visualization with empty DataFrame."""
        correlation_df = pd.DataFrame()

        visualize_correlation(correlation_df)

        captured = capsys.readouterr()
        assert "No correlation data to visualize" in captured.out


class TestCreateBinaryIndicators:
    """Test suite for create_binary_indicators function."""

    @pytest.mark.unit
    def test_create_binary_indicators_basic(self):
        """Test creating binary indicators."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'category': ['A', 'B', 'A']
        })

        result = create_binary_indicators(df, 'category')

        assert 'category_A' in result.columns
        assert 'category_B' in result.columns
        assert result.iloc[0]['category_A'] == 1
        assert result.iloc[0]['category_B'] == 0
        assert result.iloc[1]['category_A'] == 0
        assert result.iloc[1]['category_B'] == 1

    @pytest.mark.unit
    def test_create_binary_indicators_multiple_values(self):
        """Test with multiple unique values."""
        df = pd.DataFrame({
            'event_type': ['Yoga', 'Task', 'Rest', 'Yoga', 'Task']
        })

        result = create_binary_indicators(df, 'event_type')

        assert 'event_type_Yoga' in result.columns
        assert 'event_type_Task' in result.columns
        assert 'event_type_Rest' in result.columns
        assert result['event_type_Yoga'].sum() == 2
        assert result['event_type_Task'].sum() == 2
        assert result['event_type_Rest'].sum() == 1


class TestEncodeEvents:
    """Test suite for encode_events function."""

    @pytest.mark.unit
    def test_encode_events_basic(self):
        """Test basic event encoding."""
        df = pd.DataFrame({
            'event_name': ['event1', 'event2', 'event1', 'event3']
        })

        result = encode_events(df, 'event_name')

        assert 'event_name_encoded' in result.columns
        assert result['event_name_encoded'].dtype in [np.int64, np.int32, int]
        # Same events should have same encoding
        assert result.iloc[0]['event_name_encoded'] == result.iloc[2]['event_name_encoded']

    @pytest.mark.unit
    def test_encode_events_preserves_original(self):
        """Test that original column is preserved."""
        df = pd.DataFrame({
            'event_name': ['A', 'B', 'C']
        })

        result = encode_events(df, 'event_name')

        assert 'event_name' in result.columns
        assert 'event_name_encoded' in result.columns
        assert len(result.columns) == 2


class TestGetUserIdForStream:
    """Test suite for get_user_id_for_stream function."""

    @pytest.mark.unit
    def test_get_user_id_basic(self):
        """Test basic user ID extraction."""
        log_mapping = {
            'log1': 'user1_12345',
            'log2': 'user2_67890'
        }

        stream_id = 'user1_eeg_001'
        result = get_user_id_for_stream(stream_id, log_mapping)

        assert result == 'user1'

    @pytest.mark.unit
    def test_get_user_id_with_muses(self):
        """Test user ID extraction with 'muses-' prefix."""
        log_mapping = {
            'log1': 'user1_muses-12345',
            'log2': 'user2_67890'
        }

        stream_id = 'muses-12345_eeg_001'
        result = get_user_id_for_stream(stream_id, log_mapping)

        assert result == 'user1'

    @pytest.mark.unit
    def test_get_user_id_not_found(self):
        """Test when user ID is not found."""
        log_mapping = {
            'log1': 'user1_12345'
        }

        stream_id = 'unknown_sensor_001'
        result = get_user_id_for_stream(stream_id, log_mapping)

        assert result is None


class TestGroupDatasetsByUserId:
    """Test suite for group_datasets_by_user_id function."""

    @pytest.mark.unit
    def test_group_datasets_basic(self):
        """Test basic dataset grouping by user ID."""
        datasets = {
            'eeg': pd.DataFrame({
                'stream_id': ['user1_eeg', 'user2_eeg'],
                'data': [1.0, 2.0]
            })
        }
        log_mapping = {
            'log1': 'user1_eeg',
            'log2': 'user2_eeg'
        }

        result = group_datasets_by_user_id(datasets, log_mapping)

        assert 'user1_eeg' in result.keys()
        assert 'user2_eeg' in result.keys()
        assert len(result['user1_eeg']) == 1
        assert len(result['user2_eeg']) == 1

    @pytest.mark.unit
    def test_group_datasets_multiple_streams(self):
        """Test grouping with multiple stream types."""
        datasets = {
            'eeg': pd.DataFrame({
                'stream_id': ['user1_eeg', 'user2_eeg'],
                'data': [1.0, 2.0]
            }),
            'gsr': pd.DataFrame({
                'stream_id': ['user1_gsr', 'user2_gsr'],
                'data': [3.0, 4.0]
            })
        }
        log_mapping = {
            'log1': 'user1_eeg',
            'log2': 'user2_gsr'
        }

        result = group_datasets_by_user_id(datasets, log_mapping)

        # Should have user1_eeg, user1_gsr, user2_eeg, user2_gsr
        assert len(result) > 0


class TestPlotIndividualUserSensorEvents:
    """Test suite for plot_individual_user_sensor_events function."""

    @pytest.mark.unit
    def test_plot_individual_basic(self, monkeypatch):
        """Test basic plotting for individual user."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'stream_id': ['sensor1', 'sensor1', 'sensor2'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'feature1': [10.0, 20.0, 15.0],
            'event_name': ['event1', 'event2', 'event1']
        })

        fig, title = plot_individual_user_sensor_events(data, 'feature1', 'user1')

        assert fig is not None
        assert 'user1' in title
        assert 'feature1' in title
        plt.close(fig)

    @pytest.mark.unit
    def test_plot_individual_with_selected_events(self, monkeypatch):
        """Test plotting with selected events filter."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1', 'user1'],
            'stream_id': ['sensor1', 'sensor1', 'sensor1', 'sensor1'],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1H'),
            'feature1': [10.0, 20.0, 15.0, 25.0],
            'event_name': ['event1', 'event2', 'event3', 'event1']
        })

        fig, title = plot_individual_user_sensor_events(
            data, 'feature1', 'user1', selected_event_names=['event1']
        )

        assert fig is not None
        plt.close(fig)


class TestPlotCombinedUserSensorEventsWithMarkers:
    """Test suite for plot_combined_user_sensor_events_with_markers function."""

    @pytest.mark.unit
    def test_plot_combined_basic(self, monkeypatch):
        """Test basic combined plotting."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1', 'user1'],
            'stream_id': ['sensor1', 'sensor1', 'sensor2', 'sensor2'],
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1H'),
            'feature1': [10.0, 20.0, 15.0, 25.0],
            'event_type': ['Yoga', 'Task', 'Yoga', 'Task']
        })

        fig, title = plot_combined_user_sensor_events_with_markers(data, 'feature1', 'user1')

        assert fig is not None
        assert 'user1' in title
        assert 'feature1' in title
        plt.close(fig)

    @pytest.mark.unit
    def test_plot_combined_with_event_type_filter(self, monkeypatch):
        """Test plotting with event type filter."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'stream_id': ['sensor1', 'sensor1', 'sensor1'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'feature1': [10.0, 20.0, 15.0],
            'event_type': ['Yoga', 'Task', 'Rest']
        })

        fig, title = plot_combined_user_sensor_events_with_markers(
            data, 'feature1', 'user1', selected_event_types=['Yoga', 'Task']
        )

        assert fig is not None
        plt.close(fig)


class TestPlotFeaturesForAllUsers:
    """Test suite for plot_features_for_all_users function."""

    @pytest.mark.unit
    def test_plot_all_users_with_event_types(self, monkeypatch):
        """Test plotting for all users with event types."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        # Mock save_figures to avoid file I/O
        with patch('correlate_features.save_figures') as mock_save:
            data = pd.DataFrame({
                'user_id': ['user1', 'user1', 'user2', 'user2'],
                'stream_id': ['sensor1', 'sensor1', 'sensor2', 'sensor2'],
                'timestamp': pd.date_range('2024-01-01', periods=4, freq='1H'),
                'feature1': [10.0, 20.0, 15.0, 25.0],
                'event_type': ['Yoga', 'Task', 'Yoga', 'Task'],
                'event_name': ['event1', 'event2', 'event1', 'event2']
            })

            non_feature_cols = ['user_id', 'stream_id', 'timestamp', 'event_type', 'event_name']

            plot_features_for_all_users(
                data, non_feature_cols, 'test_type', '/tmp/test',
                selected_event_types=['Yoga', 'Task']
            )

            # Should have called save_figures for each user and feature
            assert mock_save.call_count > 0

    @pytest.mark.unit
    def test_plot_all_users_with_selected_events(self, monkeypatch):
        """Test plotting for all users with selected events."""
        monkeypatch.setattr(plt, 'show', lambda: None)

        with patch('correlate_features.save_figures') as mock_save:
            data = pd.DataFrame({
                'user_id': ['user1', 'user1'],
                'stream_id': ['sensor1', 'sensor1'],
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='1H'),
                'feature1': [10.0, 20.0],
                'event_name': ['event1', 'event2']
            })

            non_feature_cols = ['user_id', 'stream_id', 'timestamp', 'event_name']

            plot_features_for_all_users(
                data, non_feature_cols, 'test_type', '/tmp/test',
                selected_events=['event1']
            )

            assert mock_save.call_count > 0


class TestSaveFigures:
    """Test suite for save_figures function."""

    @pytest.fixture
    def temp_save_path(self):
        """Create temporary directory for saving figures."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.unit
    def test_save_figures_basic(self, temp_save_path):
        """Test basic figure saving."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        figs = [fig]
        titles = ['Test Plot']

        save_figures(figs, titles, 'test_name', temp_save_path)

        # Check that directory was created
        save_dir = os.path.join(temp_save_path, 'test_name')
        assert os.path.exists(save_dir)

        # Check that file was created
        files = os.listdir(save_dir)
        assert len(files) == 1
        assert files[0].endswith('.png')
        assert 'Test_Plot' in files[0]
        plt.close(fig)

    @pytest.mark.unit
    def test_save_figures_special_characters(self, temp_save_path):
        """Test figure saving with special characters in title."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        figs = [fig]
        titles = ['Test: Plot/With\\Special*Chars?']

        save_figures(figs, titles, 'test', temp_save_path)

        save_dir = os.path.join(temp_save_path, 'test')
        files = os.listdir(save_dir)
        assert len(files) == 1
        # Special characters should be replaced with underscores
        assert ':' not in files[0]
        assert '/' not in files[0]
        assert '\\' not in files[0]
        plt.close(fig)

    @pytest.mark.unit
    def test_save_figures_multiple(self, temp_save_path):
        """Test saving multiple figures."""
        figs = []
        titles = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([i, i+1, i+2])
            figs.append(fig)
            titles.append(f'Plot {i}')

        save_figures(figs, titles, 'multi_test', temp_save_path)

        save_dir = os.path.join(temp_save_path, 'multi_test')
        files = os.listdir(save_dir)
        assert len(files) == 3

        for fig in figs:
            plt.close(fig)


class TestCalculateCrossStreamCorrelation:
    """Test suite for calculate_cross_stream_correlation function."""

    @pytest.mark.unit
    @patch('correlate_features.calculate_correlation')
    @patch('correlate_features.merge_features_with_events')
    def test_cross_stream_basic(self, mock_merge, mock_corr):
        """Test basic cross-stream correlation calculation."""
        # Setup mocks
        mock_merge.return_value = pd.DataFrame({
            'epoch': [0, 1],
            'feature1': [1.0, 2.0]
        })
        mock_corr.return_value = pd.DataFrame({'corr': [0.5]})

        grouped_datasets = {
            'user1_eeg': pd.DataFrame({
                'stream_id': ['s1', 's1'],
                'epoch': [0, 1],
                'feature1': [1.0, 2.0]
            }),
            'user1_gsr': pd.DataFrame({
                'stream_id': ['s2', 's2'],
                'epoch': [0, 1],
                'feature2': [3.0, 4.0]
            })
        }
        events_df = pd.DataFrame({
            'stream_id': ['s1', 's2'],
            'epoch': [0, 1],
            'event': ['e1', 'e2']
        })

        result = calculate_cross_stream_correlation(grouped_datasets, events_df, threshold=0.3)

        assert isinstance(result, dict)
        # Should have user1_eeg_user1_gsr key
        assert any('user1' in key for key in result.keys())
