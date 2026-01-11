"""
Main training script for the safety classifier
"""
from ai_safety_guard.models.advanced_trainer import AdvancedSafetyModelTrainer
from ai_safety_guard.data.dataset import SafetyDataset
from ai_safety_guard.utils.data_utils import create_synthetic_dataset
import torch


def main():
    print("Starting AI Safety Guard training...")
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset()
    texts, labels = zip(*dataset)
    
    # Initialize trainer
    print("Initializing advanced trainer...")
    trainer = AdvancedSafetyModelTrainer()
    
    # Prepare dataset
    print("Preparing dataset...")
    hf_dataset = trainer.prepare_dataset(list(texts), list(labels))
    
    # Split dataset into train and eval (80-20 split)
    train_test_split = hf_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Train the model
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./models/safety_model",
        num_epochs=3,
        batch_size=8
    )
    
    print("Training completed! Model saved to ./models/safety_model")


if __name__ == "__main__":
    main()