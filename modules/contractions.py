# contractions.py
# Utility for expanding contractions and normalizing pronunciations in text

import re
import os
import json
import logging

logger = logging.getLogger(__name__)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMALIZATION_DATA_DIR = os.path.join(MODULE_DIR, "normalization_data")

def load_json_map(filename: str) -> dict:
    """Loads a JSON file into a dictionary."""
    filepath = os.path.join(NORMALIZATION_DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Normalization data file not found: {filepath}. Returning empty map.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {filepath}. Returning empty map.")
        return {}

def load_word_list_from_file(filename: str) -> list:
    """Loads a list of words from a text file (one word per line)."""
    filepath = os.path.join(NORMALIZATION_DATA_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.warning(f"Word list file not found: {filepath}. Returning empty list.")
        return []

# Global variables for loaded data and compiled regex
CONTRACTIONS: dict = {}
COMMON_MISSPELLINGS: dict = {}
CUSTOM_DICTIONARY_WORDS: list = []
CONTRACTION_RE = None
MISSPELLING_RE = None
spell = None
SPELL_CHECKER_AVAILABLE = False

def _load_and_compile_normalization_data():
    """Loads all normalization data and compiles regex patterns."""
    global CONTRACTIONS, COMMON_MISSPELLINGS, CUSTOM_DICTIONARY_WORDS
    global CONTRACTION_RE, MISSPELLING_RE
    global spell, SPELL_CHECKER_AVAILABLE

    CONTRACTIONS = load_json_map("contractions_map.json")
    COMMON_MISSPELLINGS = load_json_map("common_misspellings_map.json")
    CUSTOM_DICTIONARY_WORDS = load_word_list_from_file("custom_dictionary.txt")

    if CONTRACTIONS:
        CONTRACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\b", re.IGNORECASE)
    else:
        CONTRACTION_RE = None

    if COMMON_MISSPELLINGS:
        MISSPELLING_RE = re.compile(r"\b(" + "|".join(map(re.escape, COMMON_MISSPELLINGS.keys())) + r")\b", re.IGNORECASE)
    else:
        MISSPELLING_RE = None

    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        SPELL_CHECKER_AVAILABLE = True
        if CUSTOM_DICTIONARY_WORDS:
            spell.word_frequency.load_words(CUSTOM_DICTIONARY_WORDS)
        logger.info("Normalization data and spellchecker reloaded/initialized.")
    except ImportError:
        spell = None
        SPELL_CHECKER_AVAILABLE = False
        logger.warning("Spellchecker library not found. Spell checking will be disabled.")

# Initial load when the module is imported
_load_and_compile_normalization_data()

def reload_normalization_data():
    """Public function to trigger a reload of all normalization data."""
    logger.info("Attempting to reload normalization data...")
    _load_and_compile_normalization_data()

def normalize_text(text: str) -> str:
    def replace_contraction(match):
        word = match.group(0)
        expanded = CONTRACTIONS.get(word.lower())
        if expanded:
            return expanded
        return word

    def replace_misspelling(match):
        word = match.group(0)
        corrected = COMMON_MISSPELLINGS.get(word.lower())
        return corrected if corrected else word

    if CONTRACTION_RE:
        text = CONTRACTION_RE.sub(replace_contraction, text)
    if MISSPELLING_RE:
        text = MISSPELLING_RE.sub(replace_misspelling, text)

    # Spell-checking step (optional, if library is available)
    if SPELL_CHECKER_AVAILABLE and spell:
        words = text.split()
        # Find misspelled words
        misspelled = spell.unknown(words) # type: ignore
        corrected_words = []
        for word in words:
            # If the word is misspelled, get the one `most likely` answer
            corrected_words.append(spell.correction(word) if word in misspelled and spell.correction(word) is not None else word)
        text = " ".join(corrected_words)

    # logger.debug(f"Normalized '{original_text}' to '{text}'") # Optional: for debugging
    return text
