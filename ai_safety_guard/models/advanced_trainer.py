"""
Advanced model training and fine-tuning module for the safety classifier
Includes validation, metrics, and better training configurations
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from typing import Dict, List, Optional, Tuple
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


class AdvancedSafetyModelTrainer:
    """
    Advanced class to handle training of the safety classification model with better validation and metrics
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased", 
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ):
        """
        Initialize the advanced trainer
        
        Args:
            model_name: Name of the pre-trained model to use
            num_labels: Number of classification labels
            learning_rate: Learning rate for training
            weight_decay: Weight decay for optimizer
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Store training history
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        
        Args:
            eval_pred: Predictions from the model
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
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
    
    def train(
        self, 
        train_dataset: HFDataset, 
        eval_dataset: HFDataset = None, 
        output_dir: str = "./safety-model", 
        num_epochs: int = 3, 
        batch_size: int = 16,
        early_stopping_patience: int = 2
    ) -> Dict:
        """
        Train the safety model with advanced configuration
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training results
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
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=0.1,
            report_to=None,  # Disable wandb logging
        )
        
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=callbacks if callbacks else None
        )
        
        # Start training
        train_result = trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        if eval_dataset:
            # Get eval results after training
            eval_result = trainer.evaluate()
            self.training_history['final_eval'] = eval_result
        
        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return {
            'train_result': train_result,
            'model_path': output_dir,
            'history': self.training_history
        }
    
    def evaluate_model(self, test_dataset: HFDataset) -> Dict:
        """
        Evaluate the model on a test dataset
        
        Args:
            test_dataset: Test dataset to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./temp",  # Temporary directory
                do_train=False,
                do_eval=True,
                per_device_eval_batch_size=16,
                report_to=None  # Disable logging
            ),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        eval_result = trainer.evaluate(test_dataset)
        return eval_result