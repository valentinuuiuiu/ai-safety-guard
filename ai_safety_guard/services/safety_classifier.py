"""
Main safety classifier service
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union


class SafetyClassifier:
    """
    A content safety classifier to detect potentially harmful text prompts
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the safety classifier
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For now, we'll use a model with 2 labels: safe (0) and unsafe (1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
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
        
        return {
            "text": text,
            "is_safe": predicted_class == 0,
            "confidence": confidence,
            "prediction": "safe" if predicted_class == 0 else "unsafe",
            "class_id": predicted_class
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