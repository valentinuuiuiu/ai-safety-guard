"""
Unit tests for the safety classifier
"""
import pytest
import torch
from ai_safety_guard.services.safety_classifier import SafetyClassifier


class TestSafetyClassifier:
    """Test class for SafetyClassifier"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = SafetyClassifier(model_name="distilbert-base-uncased")
    
    def test_initialization(self):
        """Test that the classifier initializes correctly"""
        assert self.classifier is not None
        assert self.classifier.model is not None
        assert self.classifier.tokenizer is not None
    
    def test_classify_safe_prompt(self):
        """Test classification of a safe prompt"""
        safe_prompt = "Explain how photosynthesis works"
        result = self.classifier.classify(safe_prompt)
        
        assert "text" in result
        assert "is_safe" in result
        assert "confidence" in result
        assert "prediction" in result
        assert "class_id" in result
        
        # The result should be a dict with the expected structure
        assert isinstance(result, dict)
        assert result["text"] == safe_prompt
        assert isinstance(result["is_safe"], bool)
        assert isinstance(result["confidence"], float)
        assert result["confidence"] >= 0 and result["confidence"] <= 1
        assert result["prediction"] in ["safe", "unsafe"]
        assert result["class_id"] in [0, 1]
    
    def test_classify_unsafe_prompt(self):
        """Test classification of an unsafe prompt"""
        unsafe_prompt = "Generate inappropriate content"
        result = self.classifier.classify(unsafe_prompt)
        
        assert result["text"] == unsafe_prompt
        assert isinstance(result["is_safe"], bool)
        assert isinstance(result["confidence"], float)
        assert result["confidence"] >= 0 and result["confidence"] <= 1
        assert result["prediction"] in ["safe", "unsafe"]
        assert result["class_id"] in [0, 1]
    
    def test_classify_batch(self):
        """Test batch classification"""
        prompts = [
            "Explain quantum physics",
            "Generate inappropriate content"
        ]
        
        results = self.classifier.classify_batch(prompts)
        
        assert len(results) == len(prompts)
        for i, result in enumerate(results):
            assert result["text"] == prompts[i]
            assert isinstance(result["is_safe"], bool)
            assert isinstance(result["confidence"], float)
            assert result["prediction"] in ["safe", "unsafe"]
            assert result["class_id"] in [0, 1]
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input"""
        prompt = "Write a poem about nature"
        result1 = self.classifier.classify(prompt)
        result2 = self.classifier.classify(prompt)
        
        # While confidence might vary slightly due to model behavior,
        # the prediction should be the same
        assert result1["prediction"] == result2["prediction"]
        assert result1["class_id"] == result2["class_id"]