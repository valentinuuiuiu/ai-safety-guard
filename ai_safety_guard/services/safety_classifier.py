"""
Main safety classifier service
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
from ..config import get_config


class SafetyClassifier:
    """
    A content safety classifier to detect potentially harmful text prompts
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the safety classifier
        
        Args:
            model_name: Name of the pre-trained model to use (optional, defaults to config)
        """
        config = get_config()
        model_name = model_name or config.model_name
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For now, we'll use a model with 2 labels: safe (0) and unsafe (1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Enhanced safety checks beyond ML model
        self.unsafe_keywords = [
            'inappropriate content', 'remove clothes', 'deepfake', 'bypass safety',
            'hate speech', 'violent imagery', 'scam people', 'manipulate someone',
            'promotes self-harm', 'generate nudes', 'sexual content', 'explicit content',
            'illegal activity', 'harmful content', 'dangerous activity'
        ]
        
        # Set default threshold from config
        self.default_threshold = config.default_threshold
        
    def classify(self, text: str) -> Dict[str, Union[float, int, str]]:
        """
        Classify a single text prompt for safety
        
        Args:
            text: The text prompt to classify
            
        Returns:
            Dictionary with classification results
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Additional keyword-based safety check
        text_lower = text.lower()
        keyword_match = any(keyword in text_lower for keyword in self.unsafe_keywords)
        
        # If keyword match found, force classification as unsafe regardless of ML prediction
        if keyword_match:
            return {
                "text": text,
                "is_safe": False,
                "confidence": 1.0,
                "prediction": "unsafe",
                "class_id": 1,
                "keyword_matched": True,
                "matched_keywords": [kw for kw in self.unsafe_keywords if kw in text_lower]
            }
        
        return {
            "text": text,
            "is_safe": predicted_class == 0,
            "confidence": confidence,
            "prediction": "safe" if predicted_class == 0 else "unsafe",
            "class_id": predicted_class,
            "keyword_matched": False,
            "matched_keywords": []
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Union[float, int, str]]]:
        """
        Classify a batch of text prompts for safety
        
        Args:
            texts: List of text prompts to classify
            
        Returns:
            List of dictionaries with classification results
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model from a specific path
        
        Args:
            model_path: Path to the trained model directory
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
    def update_keywords(self, new_keywords: List[str]):
        """
        Update the list of unsafe keywords
        
        Args:
            new_keywords: List of new keywords to consider unsafe
        """
        self.unsafe_keywords.extend(new_keywords)
        # Remove duplicates
        self.unsafe_keywords = list(set(self.unsafe_keywords))