def create_dataset(dataset_path):
    """
    Loads the unified intent dataset from a CSV file and saves it to the specified path (for compatibility).
    """
    import pandas as pd # Import pandas here, when the function is called

    df = pd.read_csv('intent_data/intent_dataset.csv', on_bad_lines='skip')
    df.to_csv(dataset_path, index=False)
    print(f"Dataset loaded from CSV and saved at {dataset_path}")
