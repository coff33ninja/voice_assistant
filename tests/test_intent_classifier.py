import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock, mock_open
import torch
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the intent classifier module
from modules.intent_classifier import (
    initialize_intent_classifier,
    intent_tokenizer,
    intent_model,
    INTENT_LABELS_MAP,
    CONFIDENCE_THRESHOLD,
)
from modules.joint_model import JointIntentSlotModel
from modules.config import INTENT_MODEL_SAVE_PATH

@pytest.fixture
def sample_csv_data():
    """Fixture providing sample CSV training data."""
    return pd.DataFrame({
        'text': [
            'hello there', 'hi how are you', 'good morning',
            'what is the weather', 'weather forecast', 'is it raining',
            'book a flight', 'reserve a table', 'make a reservation',
            'cancel my order', 'cancel booking', 'I want to cancel'
        ],
        'label': [
            'greeting', 'greeting', 'greeting',
            'weather', 'weather', 'weather',
            'booking', 'booking', 'booking',
            'cancel', 'cancel', 'cancel'
        ]
    })

@pytest.fixture
def mock_model_path(tmp_path):
    """Fixture providing a temporary model path for testing."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    # Create mock model files
    (model_dir / "config.json").write_text('{"num_labels": 4}')
    (model_dir / "pytorch_model.bin").write_text("mock_model_weights")
    (model_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "DistilBertTokenizer"}')
    (model_dir / "vocab.txt").write_text("mock_vocab")
    return str(model_dir)

@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode_plus.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    tokenizer.convert_tokens_to_ids.return_value = [1, 2, 3, 4, 5]
    return tokenizer

@pytest.fixture
def mock_joint_model():
    """Fixture providing a mock JointIntentSlotModel."""
    model = Mock(spec=JointIntentSlotModel)
    model.eval.return_value = None
    # Mock forward pass output
    mock_output = Mock()
    mock_output.intent_logits = torch.tensor([[0.1, 0.8, 0.05, 0.05]])
    mock_output.slot_logits = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]])
    model.return_value = mock_output
    model.__call__ = Mock(return_value=mock_output)
    return model

@pytest.fixture(autouse=True)
def reset_global_state():
    """Fixture to reset global state before each test."""
    global intent_tokenizer, intent_model
    original_tokenizer = intent_tokenizer
    original_model = intent_model
    yield
    intent_tokenizer = original_tokenizer
    intent_model = original_model
