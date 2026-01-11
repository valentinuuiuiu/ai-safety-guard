# AI Safety Guard

A content safety classifier to detect potentially harmful text prompts and prevent generation of inappropriate content.

## Overview

This project provides a safety classifier that can be integrated into AI systems to detect requests for generating harmful content, including but not limited to:
- Requests for generating inappropriate images
- Attempts to bypass safety measures
- Prompts with malicious intent

## Features

- Text-based safety classification using transformer models
- **Enhanced keyword-based detection** for immediate blocking of known threats
- RESTful API with FastAPI integration
- Configurable thresholds and safety parameters
- Batch processing capabilities
- Advanced training pipeline with metrics and validation
- Dynamic keyword updates
- Model loading and saving functionality
- Comprehensive testing suite

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Classification
```python
from ai_safety_guard.services.safety_classifier import SafetyClassifier

classifier = SafetyClassifier()
result = classifier.classify("your text prompt here")
print(result)  # Returns safety score and classification
```

### API Usage
```bash
# Run the API server
python run_api.py

# Or using the entry point
ai-safety-guard-api
```

Then access the API at `http://localhost:8000` with endpoints:
- `POST /classify` - Single text classification
- `POST /classify/batch` - Batch text classification

### Training
```bash
# Train the model
python train_model.py

# Or using the entry point
ai-safety-guard-train
```

## Enhanced Capabilities

### Dual-layer Safety Detection
1. **ML-based classification**: Transformer model for semantic understanding
2. **Keyword-based filtering**: Immediate blocking of known unsafe terms

### Configuration Management
The system uses a centralized configuration module (`ai_safety_guard/config.py`) that supports:
- Environment variable overrides
- Adjustable safety thresholds
- Customizable model parameters
- API server settings

### API Endpoints
- `/` - Health check
- `/classify` - Single text classification with configurable threshold
- `/classify/batch` - Batch text classification

## Project Structure

- `ai_safety_guard/` - Main package
  - `api.py` - FastAPI application
  - `config.py` - Configuration management
  - `data/` - Dataset files and preprocessing utilities
  - `models/` - Model implementations and training code
    - `trainer.py` - Basic trainer
    - `advanced_trainer.py` - Advanced trainer with metrics
  - `services/` - Main service classes and API
    - `safety_classifier.py` - Enhanced classifier with keyword detection
  - `tests/` - Unit and integration tests
  - `utils/` - Helper functions and utilities
- `demo.py` - Basic demo script
- `enhanced_demo.py` - Comprehensive demo with new features
- `train_model.py` - Training script using advanced trainer
- `run_api.py` - API server startup script
- `setup.py` - Package configuration with entry points

## Key Improvements Made

1. **Enhanced Safety Classifier**: Added keyword-based detection alongside ML model
2. **REST API**: Implemented FastAPI interface for easy integration
3. **Advanced Training**: Added metrics, validation, and early stopping
4. **Configuration Management**: Centralized settings with environment support
5. **Dynamic Updates**: Ability to update unsafe keywords at runtime
6. **Batch Processing**: Efficient handling of multiple requests
7. **Package Structure**: Proper Python package with entry points
8. **Comprehensive Demos**: Both basic and enhanced demonstration scripts

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run quick test:
```bash
python quick_test.py
```

3. Try the enhanced demo:
```bash
python enhanced_demo.py
```

4. Launch the API server:
```bash
python run_api.py
```

## Entry Points

The package includes convenient command-line entry points:
- `ai-safety-guard-api` - Starts the API server
- `ai-safety-guard-train` - Runs the training script
