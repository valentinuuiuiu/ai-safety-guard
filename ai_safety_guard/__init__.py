"""
AI Safety Guard - Content Safety Classifier

A package to detect potentially harmful text prompts and prevent generation of inappropriate content.
"""
from .services.safety_classifier import SafetyClassifier
from .models.advanced_trainer import AdvancedSafetyModelTrainer
from .config import get_config, SafetyGuardConfig

__version__ = "1.0.0"
__author__ = "AI Safety Team"

__all__ = [
    "SafetyClassifier",
    "AdvancedSafetyModelTrainer", 
    "get_config",
    "SafetyGuardConfig"
]