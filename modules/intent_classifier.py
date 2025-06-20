import asyncio
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)
from .config import MODEL_SAVE_PATH

intent_tokenizer = None
intent_model = None
intent_classifier_pipeline = None

INTENT_LABELS_MAP = {
    0: "set_reminder",
    1: "calendar_query",
    2: "get_weather",
    3: "general_query",
    4: "list_reminders",
    5: "retrain_model",
    6: "cancel_task",}

CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to accept a classified intent

def initialize_intent_classifier():
    global intent_tokenizer, intent_model, intent_classifier_pipeline
    print("Initializing Intent Classifier...")
    try:
        intent_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_SAVE_PATH)
        intent_model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        intent_classifier_pipeline = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)
        print("Intent Classifier initialized.")
    except Exception as e:
        print(f"Error initializing Intent Classifier from {MODEL_SAVE_PATH}: {e}")
        print("Ensure the model has been trained and saved correctly via setup_assistant.py.")
        raise

async def detect_intent_async(text: str) -> str:
    if intent_classifier_pipeline is None:
        raise RuntimeError("Intent Classifier not initialized. Call initialize_intent_classifier() first.")

    # The pipeline can return a list of dicts or a single dict
    classifier_output = await asyncio.to_thread(intent_classifier_pipeline, text)

    # Ensure results is a list
    results = list(classifier_output) if isinstance(classifier_output, list) else [classifier_output]

    if not results:
        print("Warning: Intent classifier returned no results.")
        return "general_query"

    top_result = results[0]  # Take the first result (highest probability)

    label_str = top_result.get("label") if isinstance(top_result, dict) else None
    score = top_result.get("score") if isinstance(top_result, dict) else 0.0

    if label_str is None:
        print("Warning: Could not parse label_str from classifier output.")
        return "general_query"

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
        return "general_query"

    try:
        # Expecting labels like 'LABEL_0', 'LABEL_1', etc.
        label_idx = int(label_str.split("_")[1])
        detected_intent = INTENT_LABELS_MAP.get(label_idx)
        if detected_intent:
            print(f"Detected intent: '{detected_intent}' with score {score:.4f}")
            return detected_intent
        else:
            print(f"Warning: Parsed label_idx {label_idx} not in INTENT_LABELS_MAP.")
            return "general_query"
    except (IndexError, ValueError, TypeError):
        print(
            f"Warning: Could not parse label_str '{label_str}' or map it to a known intent."
        )
        return "general_query"
