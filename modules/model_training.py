from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
import os

def fine_tune_model(dataset_path, model_save_path):
    # Load dataset from unified CSV
    loaded_dataset = load_dataset("csv", data_files=dataset_path)
    from datasets import DatasetDict, Dataset as HFDataset
    if isinstance(loaded_dataset, DatasetDict):
        dataset = loaded_dataset["train"]
    elif isinstance(loaded_dataset, HFDataset):
        dataset = loaded_dataset
    else:
        raise ValueError("Loaded dataset is not a supported HuggingFace Dataset type.")

    # Dynamically generate label_map from CSV
    df = pd.read_csv(dataset_path)
    unique_labels = sorted(df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})

    # Tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=100,
        save_total_limit=2,
        logging_dir=os.path.join(model_save_path, "logs"),
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
