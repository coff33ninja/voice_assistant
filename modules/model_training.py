import os
import sys
import argparse # For parsing command-line arguments


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

    # Load the CSV to get unique labels and potentially entities in the future
    import pandas as pd # Import pandas here, just before it's used
    df = pd.read_csv(dataset_path)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Dynamically generate label_map from the 'label' column
    unique_labels = sorted(df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label_map.items()} # For model config

    dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})

    # Tokenizer and model
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map), id2label=id2label, label2id=label_map)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    # Ensure output_dir and logging_dir are created if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    logging_output_dir = os.path.join(model_save_path, "logs")
    os.makedirs(logging_output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=100,
        save_total_limit=2,
        logging_dir=logging_output_dir,
        logging_steps=50, # Log more frequently
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune
    trainer.train()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Fine-tuned model saved at {model_save_path}")
    print(f"Tokenizer saved at {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an intent classification model.")
    parser.add_argument("dataset_path", type=str, help="Path to the training dataset CSV file.")
    parser.add_argument("model_save_path", type=str, help="Path where the fine-tuned model will be saved.")
    args = parser.parse_args()

    print(f"Model training script ({__file__}) started.")
    print(f"Received dataset_path: {args.dataset_path}")
    print(f"Received model_save_path: {args.model_save_path}")

    if not os.path.isfile(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1) # Use sys.exit for script termination

    fine_tune_model(args.dataset_path, args.model_save_path)
    print("Model training script finished successfully.")
