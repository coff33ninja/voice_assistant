import asyncio
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)
from .config import INTENT_MODEL_SAVE_PATH
from typing import Tuple, Dict, Any
import pandas as pd
import os

intent_tokenizer = None
intent_model = None
intent_classifier_pipeline = None

# Dynamically generate INTENT_LABELS_MAP from CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), '../intent_data/intent_dataset.csv')
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
unique_labels = sorted(df['label'].unique())
INTENT_LABELS_MAP = {idx: label for idx, label in enumerate(unique_labels)}

CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence to accept a classified intent

def initialize_intent_classifier():
    global intent_tokenizer, intent_model, intent_classifier_pipeline
    print("Initializing Intent Classifier...")
    try:
        intent_tokenizer = DistilBertTokenizer.from_pretrained(INTENT_MODEL_SAVE_PATH)
        intent_model = DistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_SAVE_PATH)
        intent_classifier_pipeline = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)
        print("Intent Classifier initialized.")
    except Exception as e:
        print(f"Error initializing Intent Classifier from {INTENT_MODEL_SAVE_PATH}: {e}")
        print("Ensure the model has been trained and saved correctly via setup_assistant.py.")
        raise

async def detect_intent_async(text: str) -> Tuple[str, Dict[str, Any]]:
    if intent_classifier_pipeline is None:
        raise RuntimeError("Intent Classifier not initialized. Call initialize_intent_classifier() first.")

    default_intent = "general_query"
    entities: Dict[str, Any] = {} # Placeholder for entities

    # The pipeline can return a list of dicts or a single dict
    classifier_output = await asyncio.to_thread(intent_classifier_pipeline, text)

    # Ensure results is a list
    results = list(classifier_output) if isinstance(classifier_output, list) else [classifier_output]
    if not results:
        print("Warning: Intent classifier returned no results.")
        return default_intent, entities

    top_result = results[0]  # Take the first result (highest probability)

    label_str = top_result.get("label") if isinstance(top_result, dict) else None
    score = top_result.get("score") if isinstance(top_result, dict) else 0.0
    if label_str is None:
        print("Warning: Could not parse label_str from classifier output.")
        return default_intent, entities

    print(
        f"Intent classifier raw output: Label='{label_str}', Score={score:.4f} for text: '{text}'"
    )

    # Ensure score is not None before comparison
    if score is None:
        score = 0.0
    if score < CONFIDENCE_THRESHOLD:
        print(
            f"Intent score {score:.4f} below threshold {CONFIDENCE_THRESHOLD}. Falling back to general_query."
        )
        return default_intent, entities

    try:
        # Expecting labels like 'LABEL_0', 'LABEL_1', etc.
        label_idx = int(label_str.split("_")[1])
        detected_intent = INTENT_LABELS_MAP.get(label_idx)
        if detected_intent:
            print(f"Detected intent: '{detected_intent}' with score {score:.4f}")
            # The 'entities' column in the CSV is primarily for training models capable of NER.
            # The current DistilBertForSequenceClassification model does not inherently extract entities.
            # The following lookup is very brittle as it requires an exact match of the user's raw text
            # with a training phrase, which is unlikely.
            # For robust entity extraction, a dedicated NER model or a joint intent-entity model is needed.
            # For now, 'entities' remains a placeholder.
            #
            # matching_row = df[(df['text'] == text) & (df['label'] == detected_intent)]
            # if not matching_row.empty and 'entities' in matching_row.columns:
            #     entities_str = matching_row.iloc[0]['entities']
            #     if entities_str and pd.notna(entities_str) and entities_str.strip() and entities_str != '{}':
            #         try: entities = json.loads(entities_str)
            #         except json.JSONDecodeError: print(f"Could not parse entities from CSV: {entities_str}")
            return detected_intent, entities
        else:
            print(f"Warning: Parsed label_idx {label_idx} not in INTENT_LABELS_MAP.")
            return default_intent, entities
    except (IndexError, ValueError, TypeError):
        print(
            f"Warning: Could not parse label_str '{label_str}' or map it to a known intent."
        )
        return default_intent, entities
