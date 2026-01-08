"""
Script to create a sample dataset for the safety classifier
"""
import pandas as pd
from ai_safety_guard.utils.data_utils import create_synthetic_dataset
import os


def create_sample_dataset():
    """
    Create a sample dataset CSV file for training
    """
    dataset = create_synthetic_dataset()
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset, columns=['text', 'label'])
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/sample_safety_dataset.csv', index=False)
    print(f"Created sample dataset with {len(dataset)} entries")
    print(f"Safe prompts: {len([x for x in dataset if x[1] == 0])}")
    print(f"Unsafe prompts: {len([x for x in dataset if x[1] == 1])}")
    
    # Display first few entries
    print("\nFirst 5 entries:")
    print(df.head())


if __name__ == "__main__":
    create_sample_dataset()