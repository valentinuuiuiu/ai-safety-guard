# AI Safety Guard

A content safety classifier to detect potentially harmful text prompts and prevent generation of inappropriate content.

## Overview

This project provides a safety classifier that can be integrated into AI systems to detect requests for generating harmful content, including but not limited to:
- Requests for generating inappropriate images
- Attempts to bypass safety measures
- Prompts with malicious intent

## Features

- Text-based safety classification
- Integration-ready API
- Extensible model architecture
- Comprehensive testing suite

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from services.safety_classifier import SafetyClassifier

classifier = SafetyClassifier()
result = classifier.classify("your text prompt here")
print(result)  # Returns safety score and classification
```

## Project Structure

- `data/` - Dataset files and preprocessing utilities
- `models/` - Model implementations and training code
- `services/` - Main service classes and API
- `tests/` - Unit and integration tests
- `utils/` - Helper functions and utilities
```# ai-safety-guard
