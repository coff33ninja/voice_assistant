import pandas as pd

def create_dataset(dataset_path):
    """
    Loads the unified intent dataset from a CSV file and saves it to the specified path (for compatibility).
    """
    df = pd.read_csv('models/intent_dataset.csv')
    df.to_csv(dataset_path, index=False)
    print(f"Dataset loaded from CSV and saved at {dataset_path}")
