#!/usr/bin/env python3
"""
Count tokens in the 'trace' column of CSV files using tiktoken.
"""
import pandas as pd
import tiktoken
from pathlib import Path

def count_tokens_in_trace_column(file_path, tokenizer):
    """
    Count tokens in the 'trace' column of a CSV file.
    
    Args:
        file_path: Path to the CSV file
        tokenizer: Tiktoken tokenizer instance
    
    Returns:
        int: Total number of tokens in the 'trace' column
    """
    print(f"Processing file: {file_path}")
    
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(file_path)
        
        # Check if 'trace' column exists
        if 'trace' not in df.columns:
            print(f"Warning: 'trace' column not found in {file_path}")
            return 0
        
        # Get all text from the 'trace' column
        trace_texts = df['trace'].fillna('').astype(str).tolist()
        
        # Count tokens for each entry in the trace column
        total_tokens = 0
        for text in trace_texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
        
        return total_tokens
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    # Base directory (project root)
    base_dir = Path(__file__).parent.parent
    
    # Specify file paths - datasets are in the out/filtered_datasets directory
    crewmate_path = base_dir / "out" / "filtered_datasets" / "crewmate_dataset.csv"
    impostor_path = base_dir / "out" / "filtered_datasets" / "impostor_dataset.csv"
    
    # Initialize the tokenizer (using the GPT-4 encoder)
    tokenizer = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by GPT-4
    
    # Count tokens in the 'trace' column of each file
    crewmate_trace_count = count_tokens_in_trace_column(crewmate_path, tokenizer)
    impostor_trace_count = count_tokens_in_trace_column(impostor_path, tokenizer)
    
    # Report results
    print("\nResults:")
    print(f"Tokens in 'trace' column of crewmate_dataset.csv: {crewmate_trace_count}")
    print(f"Tokens in 'trace' column of impostor_dataset.csv: {impostor_trace_count}")
    print(f"Total tokens in 'trace' columns: {crewmate_trace_count + impostor_trace_count}")

if __name__ == "__main__":
    main()
