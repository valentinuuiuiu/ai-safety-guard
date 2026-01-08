"""
Data loading and preprocessing module for the safety classifier
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import os


class SafetyDataset(Dataset):
    """
    Dataset class for safety classification
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer_name: str = "distilbert-base-uncased", max_length: int = 512):
        """
        Initialize the dataset
        
        Args:
            texts: List of text prompts
            labels: List of corresponding labels (0 for safe, 1 for unsafe)
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset_from_csv(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load dataset from CSV file
    
    Args:
        file_path: Path to the CSV file with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(file_path)
    return df['text'].tolist(), df['label'].tolist()


def create_dataloader(dataset: SafetyDataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for the safety dataset
    
    Args:
        dataset: SafetyDataset instance
        batch_size: Size of batches
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)