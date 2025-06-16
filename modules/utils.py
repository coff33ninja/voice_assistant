import os

def create_directories(BASE_DIR, MODEL_SAVE_PATH):
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
