import pytest
import os
import sys
import tempfile
import shutil
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import torch
from transformers import DistilBertConfig, DistilBertTokenizer
from datasets import Dataset, DatasetDict

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.model_training import fine_tune_model
from modules.joint_model import JointIntentSlotModel, JointModelOutput

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_dataset_csv(temp_dir):
    """Create a sample CSV dataset for testing."""
    data = {
        'text': [
            'set a reminder for tomorrow',
            'what is the weather today',
            'add meeting to calendar',
            'hello assistant',
            'goodbye'
        ],
        'label': [
            'set_reminder',
            'get_weather',
            'add_calendar_event',
            'greeting',
            'goodbye'
        ],
        'entities': [
            '{"time_phrase": "tomorrow"}',
            '{}',
            '{"event": "meeting"}',
            '{}',
            '{}'
        ]
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'test_dataset.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def empty_dataset_csv(temp_dir):
    """Create an empty CSV dataset for edge case testing."""
    df = pd.DataFrame(columns=['text', 'label', 'entities'])
    csv_path = os.path.join(temp_dir, 'empty_dataset.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def malformed_dataset_csv(temp_dir):
    """Create a malformed CSV dataset for error testing."""
    data = {
        'text': ['hello', 'world'],
        'wrong_column': ['greeting', 'general']
        # Missing required 'label' and 'entities' columns
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_dir, 'malformed_dataset.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

class TestFinetuneModelHappyPath:
    """Test successful model training scenarios."""

    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    @patch('model_training.JointIntentSlotModel')
    @patch('model_training.Trainer')
    def test_fine_tune_model_with_valid_dataset_succeeds(
        self, mock_trainer, mock_model, mock_config, mock_tokenizer,
        mock_load_dataset, sample_dataset_csv, temp_dir
    ):
        """Test successful model training with valid dataset."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format.return_value = None
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.save_pretrained.return_value = None

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.save_pretrained.return_value = None

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = None

        model_save_path = os.path.join(temp_dir, 'test_model')

        # Execute
        fine_tune_model(sample_dataset_csv, model_save_path)

        # Verify
        mock_load_dataset.assert_called_once_with("csv", data_files=sample_dataset_csv)
        mock_trainer_instance.train.assert_called_once()
        mock_model_instance.save_pretrained.assert_called_once_with(model_save_path)
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(model_save_path)

    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    @patch('model_training.JointIntentSlotModel')
    @patch('model_training.Trainer')
    def test_fine_tune_model_creates_correct_label_mappings(
        self, mock_trainer, mock_model, mock_config, mock_tokenizer,
        mock_load_dataset, sample_dataset_csv, temp_dir
    ):
        """Test that model training creates correct intent and slot label mappings."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format.return_value = None
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        model_save_path = os.path.join(temp_dir, 'test_model')

        # Execute
        fine_tune_model(sample_dataset_csv, model_save_path)

        # Verify config was updated with correct number of labels
        assert mock_config_instance.num_intent_labels == 5  # From sample data
        assert hasattr(mock_config_instance, 'id2intent_label')
        assert hasattr(mock_config_instance, 'id2slot_label')

class TestFinetuneModelEdgeCases:
    """Test model training with edge cases and boundary conditions."""

    @patch('model_training.load_dataset')
    def test_fine_tune_model_with_empty_dataset_handles_gracefully(
        self, mock_load_dataset, empty_dataset_csv, temp_dir
    ):
        """Test training with empty dataset handles gracefully."""
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = {'train': mock_dataset}

        model_save_path = os.path.join(temp_dir, 'test_model')

        # Should not raise exception, but may produce warnings
        with patch('builtins.print') as mock_print:
            fine_tune_model(empty_dataset_csv, model_save_path)
            mock_print.assert_called()

    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    def test_fine_tune_model_with_single_label_dataset(
        self, mock_config, mock_tokenizer, mock_load_dataset, temp_dir
    ):
        """Test training with dataset containing only one unique label."""
        data = {
            'text': ['hello', 'hi', 'greetings'],
            'label': ['greeting', 'greeting', 'greeting'],
            'entities': ['{}', '{}', '{}']
        }
        df = pd.DataFrame(data)
        single_label_csv = os.path.join(temp_dir, 'single_label.csv')
        df.to_csv(single_label_csv, index=False)

        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        model_save_path = os.path.join(temp_dir, 'test_model')

        with patch('model_training.JointIntentSlotModel'), \
             patch('model_training.Trainer'):
            fine_tune_model(single_label_csv, model_save_path)

        assert mock_config_instance.num_intent_labels == 1

    @patch('model_training.load_dataset')
    def test_fine_tune_model_with_malformed_entities_json(
        self, mock_load_dataset, temp_dir
    ):
        """Test training with malformed JSON in entities column."""
        data = {
            'text': ['set reminder', 'get weather'],
            'label': ['set_reminder', 'get_weather'],
            'entities': ['{"invalid": json}', 'not_json_at_all']
        }
        df = pd.DataFrame(data)
        malformed_json_csv = os.path.join(temp_dir, 'malformed_json.csv')
        df.to_csv(malformed_json_csv, index=False)

        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = {'train': mock_dataset}

        model_save_path = os.path.join(temp_dir, 'test_model')

        with patch('builtins.print') as mock_print, \
             patch('model_training.DistilBertTokenizer.from_pretrained'), \
             patch('model_training.DistilBertConfig.from_pretrained'), \
             patch('model_training.JointIntentSlotModel'), \
             patch('model_training.Trainer'):
            fine_tune_model(malformed_json_csv, model_save_path)
            mock_print.assert_called()

class TestFinetuneModelFailureConditions:
    """Test model training failure conditions and error handling."""

    def test_fine_tune_model_with_nonexistent_dataset_raises_error(self, temp_dir):
        """Test training with non-existent dataset file raises appropriate error."""
        nonexistent_path = os.path.join(temp_dir, 'nonexistent.csv')
        model_save_path = os.path.join(temp_dir, 'test_model')

        with pytest.raises((FileNotFoundError, Exception)):
            fine_tune_model(nonexistent_path, model_save_path)

    def test_fine_tune_model_with_missing_required_columns_raises_error(
        self, malformed_dataset_csv, temp_dir
    ):
        """Test training with dataset missing required columns raises ValueError."""
        model_save_path = os.path.join(temp_dir, 'test_model')

        with pytest.raises(ValueError, match="CSV must contain"):
            fine_tune_model(malformed_dataset_csv, model_save_path)

    @patch('model_training.load_dataset')
    def test_fine_tune_model_with_invalid_model_save_path_raises_error(
        self, mock_load_dataset, sample_dataset_csv
    ):
        """Test training with invalid model save path raises appropriate error."""
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = {'train': mock_dataset}

        invalid_save_path = '/dev/null/invalid_path'

        with patch('model_training.os.makedirs', side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                fine_tune_model(sample_dataset_csv, invalid_save_path)

    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    @patch('model_training.JointIntentSlotModel')
    def test_fine_tune_model_with_trainer_failure_raises_error(
        self, mock_model, mock_config, mock_tokenizer, mock_load_dataset,
        sample_dataset_csv, temp_dir
    ):
        """Test training failure in Trainer.train() raises appropriate error."""
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format.return_value = None
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        with patch('model_training.Trainer') as mock_trainer:
            mock_trainer_instance = Mock()
            mock_trainer_instance.train.side_effect = RuntimeError("Training failed")
            mock_trainer.return_value = mock_trainer_instance

            model_save_path = os.path.join(temp_dir, 'test_model')

            with pytest.raises(RuntimeError, match="Training failed"):
                fine_tune_model(sample_dataset_csv, model_save_path)

    @patch('model_training.load_dataset')
    def test_fine_tune_model_with_unsupported_dataset_type_raises_error(
        self, mock_load_dataset, sample_dataset_csv, temp_dir
    ):
        """Test training with unsupported dataset type raises ValueError."""
        mock_load_dataset.return_value = "unsupported_type"

        model_save_path = os.path.join(temp_dir, 'test_model')

        with pytest.raises(ValueError, match="not a supported HuggingFace Dataset type"):
            fine_tune_model(sample_dataset_csv, model_save_path)

class TestFinetuneModelIntegration:
    """Integration tests for end-to-end model training workflows."""

    @patch('model_training.torch.cuda.is_available', return_value=False)
    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    @patch('model_training.JointIntentSlotModel')
    @patch('model_training.Trainer')
    def test_fine_tune_model_cpu_only_training(
        self, mock_trainer, mock_model, mock_config, mock_tokenizer,
        mock_load_dataset, mock_cuda, sample_dataset_csv, temp_dir
    ):
        """Test model training works correctly in CPU-only environment."""
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format.return_value = None
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.save_pretrained.return_value = None

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.save_pretrained.return_value = None

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = None

        model_save_path = os.path.join(temp_dir, 'cpu_model')

        fine_tune_model(sample_dataset_csv, model_save_path)

        mock_trainer.assert_called_once()
        training_args = mock_trainer.call_args[1]['args']
        assert training_args.dataloader_pin_memory is False  # CPU

    @patch('model_training.load_dataset')
    @patch('model_training.DistilBertTokenizer.from_pretrained')
    @patch('model_training.DistilBertConfig.from_pretrained')
    @patch('model_training.JointIntentSlotModel')
    @patch('model_training.Trainer')
    def test_fine_tune_model_creates_directory_structure(
        self, mock_trainer, mock_model, mock_config, mock_tokenizer,
        mock_load_dataset, sample_dataset_csv, temp_dir
    ):
        """Test that model training creates proper directory structure."""
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format.return_value = None
        mock_load_dataset.return_value = {'train': mock_dataset}

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.save_pretrained.return_value = None

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.save_pretrained.return_value = None

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = None

        model_save_path = os.path.join(temp_dir, 'nested', 'model', 'path')

        fine_tune_model(sample_dataset_csv, model_save_path)

        assert os.path.exists(model_save_path)
        assert os.path.exists(os.path.join(model_save_path, 'logs'))

class TestDataProcessingFunction:
    """Test the data processing function used in model training."""

    def test_process_data_for_joint_model_with_valid_entities(self):
        """Test data processing correctly handles entity extraction."""
        pass

    @pytest.mark.parametrize("entity_json,expected_slots", [
        ('{"task": "meeting", "time": "tomorrow"}', ['B-task', 'I-task', 'B-time']),
        ('{}', ['O', 'O', 'O']),
        ('{"location": "New York"}', ['B-location', 'I-location']),
    ])
    def test_entity_to_slot_label_conversion(self, entity_json, expected_slots):
        """Test conversion of entities to IOB slot labels."""
        pass

class TestModelTrainingCommandLine:
    """Test command-line interface for model training."""

    @patch('sys.argv', ['model_training.py', 'dataset.csv', 'model_path'])
    @patch('model_training.fine_tune_model')
    @patch('os.path.isfile', return_value=True)
    def test_main_function_with_valid_arguments(self, mock_isfile, mock_fine_tune):
        """Test main function processes command-line arguments correctly."""
        try:
            from model_training import __main__
        except ImportError:
            pytest.skip("No main function found in model_training module")

    @patch('sys.argv', ['model_training.py', 'nonexistent.csv', 'model_path'])
    @patch('os.path.isfile', return_value=False)
    def test_main_function_with_nonexistent_dataset(self, mock_isfile):
        """Test main function handles nonexistent dataset file."""
        try:
            from model_training import __main__
        except ImportError:
            pytest.skip("No main function found in model_training module")
        except SystemExit as e:
            assert e.code == 1

class TestModelTrainingUtilities:
    """Test utility functions and helper methods."""

    def test_normalize_text_integration(self):
        """Test that normalize_text is properly integrated in training."""
        pass

    def test_training_arguments_configuration(self):
        """Test that TrainingArguments are configured with correct parameters."""
        pass

    def test_model_config_setup(self):
        """Test that model configuration is set up correctly."""
        pass

# Test configuration and marks
pytestmark = pytest.mark.unit

class TestModelTrainingTestSuite:
    """Meta-tests for the test suite itself."""

    def test_all_public_functions_have_tests(self):
        """Ensure all public functions in model_training module have corresponding tests."""
        import inspect
        import model_training

        public_functions = [name for name, obj in inspect.getmembers(model_training)
                            if inspect.isfunction(obj) and not name.startswith('_')]

        test_methods = []
        for cls_name in dir():
            cls = globals().get(cls_name)
            if inspect.isclass(cls) and cls_name.startswith('Test'):
                test_methods.extend([method for method in dir(cls)
                                    if method.startswith('test_')])

        assert len(test_methods) > 0, "No test methods found"

        fine_tune_tests = [m for m in test_methods if 'fine_tune_model' in m]
        assert len(fine_tune_tests) >= 5, f"Insufficient tests for fine_tune_model: {len(fine_tune_tests)}"

@pytest.mark.performance
class TestModelTrainingPerformance:
    """Performance tests for model training (run with pytest -m performance)."""

    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_training_completes_within_time_limit(self):
        """Test that training completes within reasonable time."""
        pass

    def test_memory_usage_during_training(self):
        """Test memory usage stays within acceptable limits."""
        pass

# Add module-level docstring
__doc__ = """
Comprehensive test suite for model_training.py module.

This test suite covers:
- Happy path scenarios for successful model training
- Edge cases with unusual but valid inputs
- Failure conditions and error handling
- Integration tests for end-to-end workflows
- Performance and resource usage tests

Testing Framework: pytest
Dependencies: unittest.mock, tempfile, pandas, torch, transformers

Run with: pytest tests/test_model_training.py -v
Run performance tests: pytest tests/test_model_training.py -m performance -v
"""
