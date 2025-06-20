import pandas as pd
import subprocess
import sys
from collections import Counter
from modules.contractions import normalize_text, CONTRACTIONS, COMMON_MISSPELLINGS
import json
from collections import defaultdict
from modules.device_detector import (
    get_cuda_device_name_with_torch,
    detect_cuda_with_torch,
    detect_cpu_vendor
)

# Helper: Levenshtein distance (edit distance)
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Helper: English word list for filtering misspellings
try:
    import nltk
    nltk.data.find('corpora/words')
except (ImportError, LookupError):
    try:
        nltk.download('words')
    except Exception:
        pass
try:
    from nltk.corpus import words as nltk_words
    ENGLISH_WORDS = set(nltk_words.words())
except Exception:
    ENGLISH_WORDS = set()

# Ensure pyspellchecker and nltk are installed
try:
    from spellchecker import SpellChecker
except ImportError:
    print("pyspellchecker not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspellchecker"])
    from spellchecker import SpellChecker

try:
    import nltk
    from nltk.corpus import wordnet
except ImportError:
    print("nltk not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.corpus import wordnet

# Download wordnet if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

spell = SpellChecker()

# Use device_detector for device info
try:
    import torch
    cuda_available = detect_cuda_with_torch()
    device_type = "cuda" if cuda_available else "cpu"
    device_name = get_cuda_device_name_with_torch() if cuda_available else detect_cpu_vendor()
    device_obj = torch.device("cuda" if cuda_available else "cpu")
    total_mem = (
        f"{round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB"
        if cuda_available else "N/A"
    )
    gpu_info = {
        "device": device_type,
        "device_name": device_name,
        "total_memory": total_mem
    }
except Exception as e:
    gpu_info = {
        "device": "unavailable",
        "device_name": "N/A",
        "total_memory": "N/A",
        "error": str(e)
    }

print(f"\n--- Hardware Info ---\nUsing: {gpu_info['device']} | Device: {gpu_info['device_name']} | RAM: {gpu_info['total_memory']}")

# Try to load T5 for model-based paraphrasing
T5_AVAILABLE = False
t5_tokenizer = None
t5_model = None
T5_DEVICE = "cuda" if (('device_obj' in locals()) and str(device_obj) == "cuda") else "cpu"
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    t5_model = t5_model.to(torch.device(T5_DEVICE)) # type: ignore
    T5_AVAILABLE = True
except Exception as e:
    print(f"T5 model not available: {e}")

# Paths (adjust if needed)
DATASET_PATH = "intent_data/intent_dataset.csv"
RESPONSES_PATH = "intent_data/intent_responses.csv"
AUGMENTED_PATH = "intent_data/intent_dataset_augmented.csv"
MAX_PER_INTENT = 50  # Cap per intent after augmentation

# Load data
intent_df = pd.read_csv(DATASET_PATH)
responses_df = pd.read_csv(RESPONSES_PATH)

# Synonym-based paraphrasing
def synonym_replace(text):
    words = text.split()
    for i, word in enumerate(words):
        syns = wordnet.synsets(word)
        if syns and syns[0]:
            lemmas = syns[0].lemma_names() if hasattr(syns[0], 'lemma_names') else []
            for lemma in lemmas:
                if lemma.lower() != word.lower():
                    new = words.copy()
                    new[i] = lemma.replace('_', ' ')
                    return ' '.join(new)
    return None

# Model-based paraphrasing using T5
def model_paraphrase(text):
    if not (T5_AVAILABLE and t5_tokenizer is not None and t5_model is not None):
        return []
    input_text = f"paraphrase: {text}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)
    input_ids = input_ids.to(T5_DEVICE)
    outputs = t5_model.generate(
        input_ids,
        max_length=64,
        num_beams=5,
        num_return_sequences=3,
        early_stopping=True
    )
    paraphrases = set()
    for output in outputs:
        paraphrased = t5_tokenizer.decode(output, skip_special_tokens=True)
        if paraphrased.lower() != text.lower():
            paraphrases.add(paraphrased)
    return list(paraphrases)

# Helper: Generate paraphrases using contractions, misspellings, synonyms, model
def generate_paraphrases(text):
    paraphrases = set()
    sources = {}
    difficulties = {}
    paraphrases.add(text)
    sources[text] = 'original'
    difficulties[text] = 'easy'
    # Expand contractions
    norm = normalize_text(text)
    if norm != text:
        paraphrases.add(norm)
        sources[norm] = 'contraction_expand'
        difficulties[norm] = 'easy'
    # Contract (reverse) if possible
    for contraction, expanded in CONTRACTIONS.items():
        if expanded in text:
            contracted = text.replace(expanded, contraction)
            paraphrases.add(contracted)
            sources[contracted] = 'contraction_contract'
            difficulties[contracted] = 'easy'
    # Misspellings from COMMON_MISSPELLINGS
    for miss, correct in COMMON_MISSPELLINGS.items():
        if correct in text:
            misspelled = text.replace(correct, miss)
            paraphrases.add(misspelled)
            sources[misspelled] = 'misspelling_common'
            difficulties[misspelled] = 'medium'
    # Spellchecker-based misspellings (limit wild ones)
    words = text.split()
    for i, word in enumerate(words):
        misspelled = spell.unknown([word])
        if misspelled:
            correction = spell.correction(word)
            if correction and correction != word:
                new_words = words.copy()
                new_words[i] = correction
                fixed = ' '.join(new_words)
                paraphrases.add(fixed)
                sources[fixed] = 'spellchecker_correction'
                difficulties[fixed] = 'medium'
        else:
            candidates = spell.candidates(word) or set()
            for cand in candidates:
                if cand != word:
                    # Limit to edit distance <=2 and must be a real word if possible
                    if edit_distance(word, cand) > 2:
                        continue
                    if ENGLISH_WORDS and cand.lower() not in ENGLISH_WORDS:
                        continue
                    new_words = words.copy()
                    new_words[i] = cand
                    missp = ' '.join(new_words)
                    paraphrases.add(missp)
                    sources[missp] = 'spellchecker_misspelling'
                    difficulties[missp] = 'medium'
    # Synonym-based
    syn = synonym_replace(text)
    if syn and syn != text:
        paraphrases.add(syn)
        sources[syn] = 'synonym_replace'
        difficulties[syn] = 'hard'
    # Model-based (T5)
    model_syns = model_paraphrase(text)
    for model_syn in model_syns:
        if model_syn and model_syn != text:
            paraphrases.add(model_syn)
            sources[model_syn] = 'model_paraphrase'
            difficulties[model_syn] = 'hard'
    return paraphrases, sources, difficulties

# Generate new rows
new_rows = []
for idx, row in intent_df.iterrows():
    text = row['text']
    label = row['label']
    entities = row['entities']
    paraphrases, sources, difficulties = generate_paraphrases(text)
    for para in paraphrases:
        if para != text:
            new_rows.append({'text': para, 'label': label, 'entities': entities, 'source_method': sources.get(para, 'unknown'), 'difficulty_level': difficulties.get(para, 'medium')})

# Ensure every intent in responses has at least 5 examples
for intent in responses_df['intent']:
    count = (intent_df['label'] == intent).sum() + sum(1 for r in new_rows if r['label'] == intent)
    if count < 5:
        # Use response text as a template
        response_row = responses_df.loc[responses_df['intent'] == intent]
        if not response_row.empty:
            response_text = response_row['response'].values[0]
            for i in range(5 - count):
                example_text = f"Hey, can you do this: {response_text.lower()}?"
                new_rows.append({'text': example_text, 'label': intent, 'entities': '{}', 'source_method': 'template_response', 'difficulty_level': 'easy'})
        else:
            for i in range(5 - count):
                new_rows.append({'text': f"Example for {intent} {i+1}", 'label': intent, 'entities': '{}', 'source_method': 'filler', 'difficulty_level': 'easy'})

# After new_rows is built and before saving:
# Combine original and new rows, then cap per intent
all_rows = [
    {**row, 'source_method': 'original', 'difficulty_level': 'easy'} for _, row in intent_df.iterrows()
] + new_rows
intent_samples = defaultdict(list)
for row in all_rows:
    intent_samples[row['label']].append(row)
# Cap per intent
final_rows = []
for intent, samples in intent_samples.items():
    if len(samples) > MAX_PER_INTENT:
        samples = samples[:MAX_PER_INTENT]
    final_rows.extend(samples)

# Save augmented dataset
augmented_df = pd.DataFrame(final_rows)
augmented_df = augmented_df.drop_duplicates(subset=['text', 'label'])
augmented_df.to_csv(AUGMENTED_PATH, index=False)

# Logging stats
stats = {
    "original_count": len(intent_df),
    "augmented_count": len(new_rows),
    "total_after_augmentation": len(final_rows),
    "by_source_method": dict(Counter(r['source_method'] for r in new_rows)),
    "by_intent_before": dict(Counter(intent_df['label'])),
    "by_intent_after": dict(Counter([r['label'] for r in final_rows])),
    "device_info": gpu_info,
}
print("\n--- Augmentation Stats ---")
print(json.dumps(stats, indent=2))
with open("intent_data/augmentation_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

# Output sample paraphrases
print("\n--- Sample Paraphrases ---")
for row in new_rows[:5]:
    print(f"[{row['source_method']}] {row['text']}")

# Optional: run intent_validator if available
try:
    from scripts.intent_validator import validate_intents
    print("\n--- Intent Validator ---")
    valid, messages = validate_intents()
    print("Validation passed!" if valid else "Validation failed:")
    for msg in messages:
        print(msg)
except Exception:
    print("Intent validator not available or failed to run.")

# Placeholder: For future parallelization or GPU config
# To parallelize T5 or other augmentation, use concurrent.futures.ThreadPoolExecutor
# For UI preview, consider Gradio or Streamlit for interactive review
