import asyncio
import torch  # Added for model loading and tensor operations
from transformers import (
    DistilBertTokenizer,
    DistilBertConfig,  # Added for loading model config
    # DistilBertForSequenceClassification is no longer used directly
    # pipeline is no longer used
)
from .config import INTENT_MODEL_SAVE_PATH
from .joint_model import JointIntentSlotModel
from typing import Tuple, Dict, Any, Optional, cast  # Added cast
import pandas as pd
import os

intent_tokenizer = None
intent_model: Optional[JointIntentSlotModel] = None  # Type hint for clarity
# intent_classifier_pipeline is removed

# Dynamically generate INTENT_LABELS_MAP from CSV
# This remains the same, but we also need to consider id2label from model config later
CSV_PATH = os.path.join(os.path.dirname(__file__), "../intent_data/intent_dataset.csv")
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
unique_labels = sorted(df["label"].unique())
INTENT_LABELS_MAP = {
    idx: label for idx, label in enumerate(unique_labels)
}  # Used as a fallback or reference

CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence to accept a classified intent


def initialize_intent_classifier():
    global intent_tokenizer, intent_model
    print("Initializing Intent Classifier with JointIntentSlotModel...")
    try:
        intent_tokenizer = DistilBertTokenizer.from_pretrained(INTENT_MODEL_SAVE_PATH)

        # Load the configuration
        loaded_config_obj = DistilBertConfig.from_pretrained(INTENT_MODEL_SAVE_PATH)

        # Ensure config_to_pass is explicitly DistilBertConfig to satisfy Pylance
        config_to_pass: DistilBertConfig
        if isinstance(loaded_config_obj, DistilBertConfig):
            config_to_pass = loaded_config_obj
        else:
            # This branch is taken if loaded_config_obj is PretrainedConfig but not DistilBertConfig.
            # This implies the saved config might not have been strictly DistilBert,
            # or from_pretrained returned a more generic type.
            # We attempt to convert it using its dictionary representation.
            # We assume loaded_config_obj has a .to_dict() method (true for PretrainedConfig).
            if hasattr(loaded_config_obj, "to_dict") and callable(
                getattr(loaded_config_obj, "to_dict")
            ):
                # Cast the result of from_dict to assure Pylance of the type.
                # The JointIntentSlotModel will validate necessary attributes.
                config_from_dict = DistilBertConfig.from_dict(
                    loaded_config_obj.to_dict()
                )
                config_to_pass = cast(DistilBertConfig, config_from_dict)
            else:
                raise TypeError(
                    f"Loaded config (type: {type(loaded_config_obj)}) is not DistilBertConfig and cannot be converted via to_dict()."
                )

        # Instantiate the model using the loaded config
        intent_model = JointIntentSlotModel(config_to_pass)

        # Load the saved weights
        model_weights_path = os.path.join(INTENT_MODEL_SAVE_PATH, "pytorch_model.bin")
        # Validate the resolved path using realpath to prevent path traversal, including symlinks
        resolved_model_weights_path = os.path.realpath(model_weights_path)
        resolved_save_path = os.path.realpath(INTENT_MODEL_SAVE_PATH)
        if not resolved_model_weights_path.startswith(resolved_save_path):
            raise ValueError("Invalid path detected for model weights.")
        if not os.path.exists(model_weights_path):
            print(f"Error: Model weights not found at {model_weights_path}")
            print(
                "Ensure the model has been trained and saved correctly via setup_assistant.py."
            )
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

        intent_model.load_state_dict(
            torch.load(model_weights_path, map_location=torch.device("cpu"))
        )
        intent_model.eval()  # Set the model to evaluation mode

        print("Intent Classifier (JointIntentSlotModel) initialized.")
    except Exception as e:
        print(
            f"Error initializing JointIntentSlotModel from {INTENT_MODEL_SAVE_PATH}: {e}"
        )
        print(
            "Ensure the model has been trained and saved correctly via setup_assistant.py."
        )
        raise


async def detect_intent_async(text: str) -> Tuple[str, Dict[str, Any]]:
    if intent_model is None or intent_tokenizer is None:  # Updated check
        raise RuntimeError(
            "Intent Classifier (JointIntentSlotModel) not initialized. Call initialize_intent_classifier() first."
        )

    default_intent = "general_query"
    entities: Dict[str, Any] = {}

    # Tokenize the input text
    # Ensure the tokenizer is called in a way that's compatible with asyncio (if it involves blocking I/O)
    # For DistilBertTokenizer, direct call is usually fine.
    inputs = intent_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )  # Max length from training

    # Ensure model is on the correct device (e.g., CPU) if not already
    # inputs = {k: v.to(intent_model.device) for k, v in inputs.items()} # Assuming model has a device attribute

    # Get model predictions
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = await asyncio.to_thread(intent_model, **inputs)
        intent_logits = outputs.intent_logits
        slot_logits = outputs.slot_logits

    # Process Intent Logits
    intent_probabilities = torch.softmax(intent_logits, dim=-1)

    # Get both values and indices from torch.max
    intent_predictions = torch.max(intent_probabilities, dim=-1)
    confidence_score_tensor = intent_predictions.values
    predicted_intent_id_tensor = intent_predictions.indices

    confidence_score = confidence_score_tensor.item()  # Should be float
    # .item() on a LongTensor (like indices) should return an int
    predicted_intent_id = predicted_intent_id_tensor.item()

    if confidence_score < CONFIDENCE_THRESHOLD:
        print(
            f"Intent score {confidence_score:.4f} below threshold {CONFIDENCE_THRESHOLD}. Falling back to {default_intent}."
        )
        return default_intent, entities

    predicted_intent_id_int = int(
        predicted_intent_id
    )  # Ensure it's an int for dict keys
    # Use the model's id2label for intent (if available from config)
    detected_intent = (
        intent_model.config.id2intent_label.get(predicted_intent_id_int, default_intent)
        if hasattr(intent_model.config, "id2intent_label")
        else INTENT_LABELS_MAP.get(predicted_intent_id_int, default_intent)
    )

    print(
        f"Detected intent: '{detected_intent}' with score {confidence_score:.4f} for text: '{text}'"
    )

    # Process Slot Logits
    predicted_slot_ids = torch.argmax(slot_logits, dim=-1)
    input_ids_list = inputs["input_ids"].squeeze().tolist()  # Get token IDs as a list
    tokens = intent_tokenizer.convert_ids_to_tokens(input_ids_list)

    # Get id2slot_label from model config (should have been saved during training)
    id2slot_label = (
        intent_model.config.id2slot_label
        if hasattr(intent_model.config, "id2slot_label")
        else {}
    )

    current_entity_value = ""
    current_entity_type = ""
    in_entity = False

    for token_idx, (token, predicted_slot_id_tensor) in enumerate(
        zip(tokens, predicted_slot_ids.squeeze())
    ):
        if token in [
            intent_tokenizer.cls_token,
            intent_tokenizer.sep_token,
            intent_tokenizer.pad_token,
        ]:
            continue  # Skip special tokens

        predicted_slot_label = id2slot_label.get(
            int(predicted_slot_id_tensor.item()), "O"
        )

        if predicted_slot_label.startswith("B-"):
            if (
                in_entity
            ):  # If already capturing an entity, store it before starting new one
                if current_entity_type and current_entity_value:
                    entities[current_entity_type] = (
                        entities.get(current_entity_type, "")
                        + current_entity_value.strip()
                        + " "
                    )  # Append if multiple
                    entities[current_entity_type] = entities[
                        current_entity_type
                    ].strip()  # Clean up trailing space
            current_entity_value = ""  # Reset for new entity
            current_entity_type = predicted_slot_label[2:]  # Get type from B-TAG
            # Handle token reconstruction (simple concatenation, may need improvement for subwords)
            current_entity_value += (
                token.replace("##", "")
                if "##" not in token
                else token.replace("##", "")
            )
            in_entity = True
        elif predicted_slot_label.startswith("I-"):
            if in_entity and current_entity_type == predicted_slot_label[2:]:
                current_entity_value += (
                    " " + token.replace("##", "")
                    if not token.startswith("##")
                    else token.replace("##", "")
                )
            elif in_entity and current_entity_type != predicted_slot_label[2:]:
                # Mismatch I-TAG, current entity ends. Start a new one if this is a B-TAG (handled by B- logic)
                if current_entity_type and current_entity_value:
                    entities[current_entity_type] = (
                        entities.get(current_entity_type, "")
                        + current_entity_value.strip()
                        + " "
                    )
                    entities[current_entity_type] = entities[
                        current_entity_type
                    ].strip()
                in_entity = False  # Reset
                current_entity_value = ""
                current_entity_type = ""
                # If this I-TAG is for a new entity type but without a B-TAG, treat as O or start new B- if logic allows
                # For now, we effectively treat it as O by resetting and not adding to current_entity_value
        else:  # O tag or unexpected tag
            if in_entity:  # Entity ended with O tag
                if current_entity_type and current_entity_value:
                    entities[current_entity_type] = (
                        entities.get(current_entity_type, "")
                        + current_entity_value.strip()
                        + " "
                    )
                    entities[current_entity_type] = entities[
                        current_entity_type
                    ].strip()
                in_entity = False
                current_entity_value = ""
                current_entity_type = ""

    # After loop, check if an entity was still being processed
    if in_entity and current_entity_type and current_entity_value:
        entities[current_entity_type] = (
            entities.get(current_entity_type, "") + current_entity_value.strip() + " "
        )
        entities[current_entity_type] = entities[current_entity_type].strip()

    # Refine entity values (remove leading/trailing spaces from concatenated parts)
    for key in entities:
        entities[key] = (
            entities[key].strip().replace(" ##", "").replace("##", "")
        )  # General cleanup for subwords

    print(f"Extracted Entities: {entities}")

    return detected_intent, entities
