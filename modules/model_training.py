import os
import sys
from modules.contractions import normalize_text  # Import the normalize_text function
import argparse  # For parsing command-line arguments

import json  # Import json for entity parsing
import torch  # Import torch for model definition and device check
from typing import cast  # Import cast for type casting
# nn is no longer directly used here after moving JointIntentSlotModel
from transformers import (  # type: ignore
    DistilBertTokenizerFast as DistilBertTokenizer,  # Use the Fast Tokenizer
    Trainer,
    TrainingArguments,
    # DistilBertModel is now imported in joint_model.py
    DistilBertConfig,
)  # Import necessary HF components
from .joint_model import (
    JointIntentSlotModel,
)  # Import the refactored model


def fine_tune_model(dataset_path, model_save_path):
    print(f"Starting fine-tuning process... [pid={os.getpid()}]")
    print(f"Using dataset: {dataset_path}")
    print(f"Saving model to: {model_save_path}")

    # Load dataset from unified CSV
    # Import datasets components here, when needed
    from datasets import load_dataset, DatasetDict, Dataset as HFDataset

    loaded_dataset = load_dataset("csv", data_files=dataset_path)
    if isinstance(loaded_dataset, DatasetDict):
        dataset = loaded_dataset["train"]
    elif isinstance(loaded_dataset, HFDataset):
        dataset = loaded_dataset
    else:
        raise ValueError("Loaded dataset is not a supported HuggingFace Dataset type.")

    # Load the CSV to get unique labels and entity types
    import pandas as pd  # Import pandas here, just before it's used

    df = pd.read_csv(dataset_path)
    if (
        "label" not in df.columns
        or "text" not in df.columns
        or "entities" not in df.columns
    ):
        raise ValueError("CSV must contain 'text', 'label', and 'entities' columns.")

    # Dynamically generate label_map from the 'label' column
    unique_intent_labels = sorted(df["label"].unique())
    intent_label_map = {label: idx for idx, label in enumerate(unique_intent_labels)}
    id2intent_label = {
        idx: label for label, idx in intent_label_map.items()
    }  # For model config

    # Dynamically generate slot_label_map from the 'entities' column
    unique_entity_types = set()
    for entities_str in df["entities"].dropna():
        try:
            entities = json.loads(entities_str)
            for entity_type in entities.keys():
                unique_entity_types.add(entity_type)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse entities JSON: {entities_str}")
            continue  # Skip malformed JSON

    # Create IOB tags
    slot_labels_list = ["O"]  # 'O' is always the first label
    for entity_type in sorted(list(unique_entity_types)):
        slot_labels_list.append(f"B-{entity_type}")
        slot_labels_list.append(f"I-{entity_type}")

    slot_label_map = {label: idx for idx, label in enumerate(slot_labels_list)}
    id2slot_label = {idx: label for label, idx in slot_label_map.items()}

    print(f"Intent Labels Map: {intent_label_map}")
    print(f"Slot Labels Map: {slot_label_map}")

    # Normalize the 'text' column before further processing
    print("Normalizing text data in the dataset...")
    dataset = dataset.map(
        lambda x: {"text": normalize_text(x["text"]) if x["text"] else ""}
    )

    # Map intent labels to IDs
    dataset = dataset.map(
        lambda x: {"label": intent_label_map[x["label"]]}
    )  # Corrected variable name

    # Data processing function to tokenize and create slot labels
    import re  # Import re for regex matching

    def process_data_for_joint_model(examples):
        # Tokenize the text
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,  # Use the same max_length as in current training
            return_offsets_mapping=True,  # Needed to map character indices to token indices
        )

        # Get intent label ID
        # 'examples["label"]' is already the integer ID from the previous .map() operation
        intent_label_id = examples["label"]

        # Initialize slot labels with the ID for 'O' (Outside)
        # The size should match the number of tokens after padding/truncation
        slot_labels = [-100] * len(
            tokenized_inputs["input_ids"]
        )  # Use -100 for ignored tokens (padding, special tokens)

        # Process entities for slot labeling
        entities_json_str = examples.get("entities", "{}")
        entities = json.loads(entities_json_str) if entities_json_str else {}

        offset_mapping = tokenized_inputs["offset_mapping"]
        sequence_ids = (
            tokenized_inputs.sequence_ids()
        )  # Helps identify special tokens ([CLS], [SEP])

        # Iterate through entity types and their values
        for entity_type, entity_value in entities.items():
            entity_value_str = str(entity_value)
            # Find all occurrences of the entity value string in the original text
            # This is a simple approach; more robust methods might use fuzzy matching or regex
            for match in re.finditer(re.escape(entity_value_str), examples["text"]):
                start_char, end_char = match.span()

                # Map character span to token span using offset_mapping and sequence_ids
                token_indices_in_entity = []
                for token_index, (char_start, char_end) in enumerate(offset_mapping):
                    if (
                        sequence_ids[token_index] is None
                        or sequence_ids[token_index] != 0
                    ):  # Only consider tokens from the original sequence (index 0)
                        continue

                    # Check for overlap: [char_start, char_end) overlaps with [start_char, end_char)
                    if max(char_start, start_char) < min(char_end, end_char):
                        token_indices_in_entity.append(token_index)

                if token_indices_in_entity:
                    # Apply IOB tags
                    b_tag = f"B-{entity_type}"
                    i_tag = f"I-{entity_type}"

                    if b_tag not in slot_label_map or i_tag not in slot_label_map:
                        print(
                            f"Warning: Slot tags '{b_tag}' or '{i_tag}' not found in slot_label_map. Skipping entity: {entity_type}"
                        )
                        continue  # Skip this entity if tags aren't in the map

                    # Ensure indices are within bounds and not special tokens
                    valid_indices = [
                        idx for idx in token_indices_in_entity if sequence_ids[idx] == 0
                    ]  # Re-check sequence_ids

                    if valid_indices:
                        slot_labels[valid_indices[0]] = slot_label_map[b_tag]
                        for token_index in valid_indices[1:]:
                            slot_labels[token_index] = slot_label_map[i_tag]

        # The tokenizer output includes input_ids, attention_mask, etc.
        # Add the intent_label_id and slot_labels to the output dictionary.
        tokenized_inputs["intent_labels"] = intent_label_id
        tokenized_inputs["slot_labels"] = slot_labels

        # Remove the offset_mapping as it's not needed for model input
        del tokenized_inputs["offset_mapping"]
        # Remove sequence_ids if it was added by the tokenizer
        if "sequence_ids" in tokenized_inputs:
            del tokenized_inputs["sequence_ids"]

        return tokenized_inputs

    # Tokenizer and model initialization
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  # type: ignore

    # Create model config with the number of labels for both tasks
    # Explicitly cast the result to DistilBertConfig to satisfy Pylance
    config: DistilBertConfig = cast(
        DistilBertConfig, DistilBertConfig.from_pretrained("distilbert-base-uncased")
    )
    config.num_intent_labels = len(intent_label_map)
    config.num_slot_labels = len(slot_label_map)
    # Optional: Store label maps in config for easier loading later
    config.id2intent_label = id2intent_label
    config.id2slot_label = id2slot_label
    config.label2id = intent_label_map  # Save intent map for loading in classifier
    config.slot_label2id = slot_label_map  # Save slot map for loading in classifier

    model = JointIntentSlotModel(config)

    # Apply the data processing function
    # Use batched=True for efficiency, but the processing function needs to handle lists of examples
    # For simplicity here, let's process one example at a time (batched=False or remove batched=True)
    # A batched version of process_data_for_joint_model is more complex.
    # Let's process one by one for clarity in this example.
    dataset = dataset.map(process_data_for_joint_model)

    # Training arguments
    # Ensure output_dir and logging_dir are created if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    logging_output_dir = os.path.join(model_save_path, "logs")
    os.makedirs(logging_output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=model_save_path,  # Model and tokenizer will be saved here
        num_train_epochs=5,  # Increased epochs
        learning_rate=2e-5,  # Explicitly set learning rate
        per_device_train_batch_size=16,  # Increased batch size
        weight_decay=0.01,  # Added weight decay
        save_strategy="epoch",  # Save at the end of each epoch
        save_total_limit=2,
        dataloader_pin_memory=torch.cuda.is_available(),  # Set based on CUDA availability
        logging_dir=logging_output_dir,
        logging_strategy="steps",  # Log based on steps
        logging_steps=10,  # Log more frequently
        # To enable evaluation during training, you would add:
        # evaluation_strategy="epoch",
        # load_best_model_at_end=True, # If using evaluation
    )

    # Set the format of the dataset to PyTorch tensors
    # Ensure all required columns are included
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "intent_labels", "slot_labels"],
    )

    # Log the type of the dataset to be passed to the Trainer for verification
    print(f"DEBUG: Type of dataset being passed to Trainer: {type(dataset)}")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune
    print("Starting model training...")
    trainer.train()
    model.save_pretrained(model_save_path)  # type: ignore
    tokenizer.save_pretrained(model_save_path)
    print(f"Fine-tuned model saved at {model_save_path}")
    print(f"Tokenizer saved at {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune an intent classification model."
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the training dataset CSV file."
    )
    parser.add_argument(
        "model_save_path",
        type=str,
        help="Path where the fine-tuned model will be saved.",
    )
    args = parser.parse_args()

    print(f"Model training script ({__file__}) started.")
    print(f"Received dataset_path: {args.dataset_path}")
    print(f"Received model_save_path: {args.model_save_path}")

    if not os.path.isfile(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1)  # Use sys.exit for script termination

    fine_tune_model(args.dataset_path, args.model_save_path)
    print("Model training script finished successfully.")
