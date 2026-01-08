"""
Unit tests for data utilities
"""
import pytest
from ai_safety_guard.utils.data_utils import create_synthetic_dataset, clean_text


class TestDataUtils:
    """Test class for data utilities"""
    
    def test_create_synthetic_dataset(self):
        """Test that synthetic dataset is created correctly"""
        dataset = create_synthetic_dataset()
        
        assert len(dataset) > 0
        assert all(len(item) == 2 for item in dataset)  # Each item should be (text, label)
        assert all(isinstance(text, str) for text, label in dataset)  # Text should be string
        assert all(label in [0, 1] for text, label in dataset)  # Label should be 0 or 1
        
        # Check that we have both safe and unsafe examples
        safe_count = sum(1 for text, label in dataset if label == 0)
        unsafe_count = sum(1 for text, label in dataset if label == 1)
        
        assert safe_count > 0
        assert unsafe_count > 0
    
    def test_clean_text(self):
        """Test text cleaning function"""
        original_text = "  This   is  a   test   string  \n\t with   extra   spaces  "
        cleaned_text = clean_text(original_text)
        
        # Should remove extra spaces
        assert "  " not in cleaned_text  # No double spaces
        assert cleaned_text.startswith("This")
        assert cleaned_text.endswith("spaces")
        assert "\n" not in cleaned_text
        assert "\t" not in cleaned_text
        
        # Should preserve the basic meaning
        assert "test" in cleaned_text
        assert "string" in cleaned_text