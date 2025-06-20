import os
import shutil
from datetime import datetime

AUGMENTED_FILES = [
    "intent_data/intent_dataset_augmented.csv",
    "intent_data/augmentation_stats.json",
    # Add more files if needed
]

ARCHIVE_DIR = "intent_data/augmented_checkpoints"

def archive_augmented_files():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for file_path in AUGMENTED_FILES:
        if os.path.exists(file_path):
            base = os.path.basename(file_path)
            archive_path = os.path.join(ARCHIVE_DIR, f"{timestamp}_{base}")
            shutil.move(file_path, archive_path)
            print(f"Archived {file_path} -> {archive_path}")
        else:
            print(f"File not found (skipped): {file_path}")

if __name__ == "__main__":
    archive_augmented_files()
    print("Augmented data files archived. Ready for fresh augmentation.")
