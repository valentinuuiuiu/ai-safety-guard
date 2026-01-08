"""
Model training and fine-tuning module for the safety classifier
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from typing import Dict, List
import os


class SafetyModelTrainer:
    """
    Class to handle training of the safety classification model
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):
        """
        Initialize the trainer
        
        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> HFDataset:
        """
        Prepare dataset for training
        
        Args:
            texts: List of text prompts
            labels: List of corresponding labels
            
        Returns:
            HuggingFace Dataset
        """
        dataset_dict = {
            'text': texts,
            'labels': labels
        }
        dataset = HFDataset.from_dict(dataset_dict)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True, 
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, train_dataset: HFDataset, eval_dataset: HFDataset = None, 
              output_dir: str = "./safety-model", num_epochs: int = 3, 
              batch_size: int = 16) -> None:
        """
        Train the safety model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)