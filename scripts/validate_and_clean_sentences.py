import pandas as pd
import language_tool_python
import os
from tqdm import tqdm # For progress bar
import argparse # For command-line arguments

tqdm.pandas()

def clean_sentence(text: str) -> str:
    """
    Cleans a single sentence using language_tool_python.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception as e:
        print(f"Warning: Error cleaning sentence '{text[:50]}...': {e}")
        return text

def main():
    parser = argparse.ArgumentParser(description="Validate and auto-correct sentences in a CSV dataset.")
    parser.add_argument("--input", default="intent_data/intent_dataset.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--output", default="intent_data/intent_dataset_cleaned.csv",
                        help="Path to save the cleaned CSV file.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show sample of changes.")
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        print(f"Input CSV '{input_csv}' is empty. No processing needed.")
        pd.DataFrame(columns=['text']).to_csv(output_csv, index=False)
        return
    except Exception as e:
        print(f"Error loading CSV file '{input_csv}': {e}")
        return

    if 'text' not in df.columns:
        raise ValueError(f"Required 'text' column not found in {input_csv}")

    df['text'] = df['text'].astype(str)

    global tool
    tool = language_tool_python.LanguageTool('en-US')

    print(f"Running sentence validation and auto-correction on {len(df)} sentences...")
    df['text_cleaned'] = df['text'].progress_apply(clean_sentence)

    changed_df = df[df['text'] != df['text_cleaned']]
    num_changes = len(changed_df)

    if args.verbose:
        print(f"\n--- Sample of {min(5, num_changes)} changes ---")
        for i, row in changed_df.head(5).iterrows():
            print(f"Original: {row['text']}\nCleaned:  {row['text_cleaned']}\n---")
        if num_changes > 5:
            print(f"...and {num_changes - 5} more changes.")

    print(f"\nCompleted. Total sentences processed: {len(df)}")
    print(f"Number of sentences corrected: {num_changes} ({num_changes/len(df):.2%})")

    cols = list(df.columns)
    if 'text_cleaned' in cols:
        df['text'] = df['text_cleaned']
        df = df.drop(columns=['text_cleaned'])
    df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
