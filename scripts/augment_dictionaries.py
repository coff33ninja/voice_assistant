import json
from pathlib import Path

# Paths for original and augmented dictionaries
DICT_DIR = Path(__file__).parent.parent / "modules" / "normalization_data"
AUGMENTED_SUFFIX = "_augmented.json"

# List of dictionary files to augment (add more as needed)
DICT_FILES = [
    "contractions.json",
    "misspellings.json",
    "synonyms.json",
    "normalization.json"
]

def augment_dictionary(orig_data):
    # Placeholder: Add real augmentation logic here
    # For now, just return a copy (no-op)
    augmented = orig_data.copy()
    # TODO: Add real augmentation logic
    return augmented

def main():
    stats = {}
    merged_dicts = {}
    for dict_file in DICT_FILES:
        orig_path = DICT_DIR / dict_file
        aug_path = DICT_DIR / (dict_file.replace(".json", AUGMENTED_SUFFIX))
        orig_data = None
        aug_data = None
        if orig_path.exists():
            with open(orig_path, "r", encoding="utf-8") as f:
                orig_data = json.load(f)
            aug_data = augment_dictionary(orig_data)
            with open(aug_path, "w", encoding="utf-8") as f:
                json.dump(aug_data, f, indent=2, ensure_ascii=False)
            stats[dict_file] = {
                "original_count": len(orig_data),
                "augmented_count": len(aug_data),
                "augmented_path": str(aug_path)
            }
            # Provide a merged view (augmented overlays original)
            if isinstance(orig_data, dict) and isinstance(aug_data, dict):
                merged = orig_data.copy()
                merged.update(aug_data)
                merged_dicts[dict_file] = merged
            else:
                merged_dicts[dict_file] = aug_data if aug_data else orig_data
        else:
            stats[dict_file] = {"error": "Original file not found"}
    print("Dictionary augmentation complete. Stats:")
    print(json.dumps(stats, indent=2))
    # Optionally, save merged view for downstream use
    merged_path = DICT_DIR / "merged_dictionaries.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_dicts, f, indent=2, ensure_ascii=False)
    print(f"Merged dictionaries saved to {merged_path}")

if __name__ == "__main__":
    main()
