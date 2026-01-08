"""
Unit tests for the dataset module
"""
import pytest
import torch
from ai_safety_guard.data.dataset import SafetyDataset
from ai_safety_guard.utils.data_utils import create_synthetic_dataset


class TestSafetyDataset:
    """Test class for SafetyDataset"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a small test dataset
        dataset = create_synthetic_dataset()[:10]  # Use first 10 items
        self.texts, self.labels = zip(*dataset)
        self.dataset = SafetyDataset(list(self.texts), list(self.labels))
    
    def test_dataset_length(self):
        """Test that dataset length matches input length"""
        assert len(self.dataset) == len(self.texts)
        assert len(self.dataset) == len(self.labels)
    
    def test_getitem_structure(self):
        """Test that __getitem__ returns the correct structure"""
        item = self.dataset[0]
        
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item
        
        # Check tensor properties
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['label'], torch.Tensor)
        
        # Check shapes
        assert item['input_ids'].dim() == 1  # 1D tensor
        assert item['attention_mask'].dim() == 1  # 1D tensor
        assert item['label'].dim() == 0  # 0D tensor (scalar)
    
    def test_getitem_values(self):
        """Test that __getitem__ returns correct values"""
        for i in range(min(3, len(self.dataset))):  # Test first 3 items
            item = self.dataset[i]
            
            # Check that label matches expected
            expected_label = torch.tensor(self.labels[i], dtype=torch.long)
            assert item['label'].item() == expected_label.item()
            
            # Check that tensors have expected properties
            assert item['input_ids'].shape[0] == 512  # max_length
            assert item['attention_mask'].shape[0] == 512  # max_length