"""
Configuration module for AI Safety Guard
Manages settings, hyperparameters, and model configurations
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SafetyGuardConfig:
    """
    Configuration class for AI Safety Guard
    """
    # Model settings
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 512
    
    # Training settings
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    early_stopping_patience: int = 2
    
    # Safety settings
    default_threshold: float = 0.5
    enable_keyword_check: bool = True
    
    # Paths
    model_save_path: str = "./models/safety_model"
    data_path: str = "./data"
    log_path: str = "./logs"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    def update_from_env(self):
        """
        Update configuration from environment variables
        """
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.num_epochs = int(os.getenv("NUM_EPOCHS", self.num_epochs))
        self.batch_size = int(os.getenv("BATCH_SIZE", self.batch_size))
        self.learning_rate = float(os.getenv("LEARNING_RATE", self.learning_rate))
        self.model_save_path = os.getenv("MODEL_SAVE_PATH", self.model_save_path)
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", self.api_port))
        self.default_threshold = float(os.getenv("DEFAULT_THRESHOLD", self.default_threshold))
        
        return self


# Global config instance
config = SafetyGuardConfig().update_from_env()


def get_config() -> SafetyGuardConfig:
    """
    Get the global configuration instance
    
    Returns:
        SafetyGuardConfig: The configuration instance
    """
    return config