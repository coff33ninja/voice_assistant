# greeting_module.py
# Contains greeting and goodbye message variations and intent logic

import random
import pandas as pd
import os

# Attempt to load greeting/goodbye phrases from the unified intent dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../models/intent_dataset.csv')

def _load_variations(label):
    try:
        df = pd.read_csv(DATASET_PATH)
        phrases = df[df['label'] == label]['text'].tolist()
        return phrases if phrases else None
    except Exception:
        return None

greeting_variations = _load_variations('greeting') or [
    "Hello! How can I help you?",
    "Hi there! What can I do for you today?",
    "Greetings! How may I assist you?",
    "Hey! How can I be of service?",
    "Good day! What would you like to do?"
]

goodbye_variations = _load_variations('goodbye') or [
    "Goodbye! Have a great day!",
    "See you later!",
    "Take care!",
    "Bye! If you need anything, just call me again.",
    "Farewell!"
]

def get_greeting():
    return random.choice(greeting_variations)

def get_goodbye():
    return random.choice(goodbye_variations)
