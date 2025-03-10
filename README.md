# MultiTask NLP

A PyTorch-based framework for training and evaluating transformer models on multiple NLP tasks, specifically text classification and named entity recognition (NER).

## Project Overview

This project provides a flexible framework for:

1. **Text Classification**: Sentiment analysis using the IMDB dataset.
2. **Named Entity Recognition (NER)**: Entity tagging using the CoNLL-2003 dataset.
3. **Sentence Embeddings**: Custom sentence transformer implementation for generating semantic text embeddings.

The codebase is designed to be modular, allowing models to be trained on individual tasks or jointly on multiple tasks.

## Features

- Task-specific data loading and preprocessing pipelines
- Custom MultiTaskTransformer model architecture that supports:
  - Text classification
  - Named entity recognition (NER)
  - Sentence embeddings
- Comprehensive training and evaluation utilities
- Stratified data splitting for balanced training
- Weighted loss functions for handling class imbalance in NER
- Visualization tools for sentence embeddings and similarity analysis

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+

### Setup

This project uses Poetry for dependency management:

```bash
# Install dependencies using Poetry
poetry install
```

## Project Structure

```
.
├── LICENSE
├── README.md
├── core
│   ├── __init__.py
│   ├── dataloader.py        # Dataset loading and preprocessing utilities
│   ├── multi_task.py        # MultiTaskTransformer model architecture
│   ├── sentence_transformer.py  # SentenceTransformer implementation
│   ├── test_classification.py   # Testing utilities for classification
│   ├── train.py             # Training and evaluation functions
│   └── train_models.py      # Model training script
├── models                   # Directory for saved model checkpoints
│   ├── classification_model.pt
│   └── ner_model.pt
├── poetry.lock
├── pyproject.toml
└── test
    └── test_sentence_transformer.py  # Tests for SentenceTransformer
```

## Usage

### Training Models

#### Text Classification

```bash
# launch your environment
poetry shell
```

```bash
python -m core.train_models --task classification --model_name bert-base-uncased --max_samples 10000 --batch_size 32 --epochs 5 --lr 2e-5
```

#### Named Entity Recognition

```bash
python -m core.train_models --task ner --model_name bert-base-uncased --max_samples 10000 --batch_size 16 --epochs 5 --lr 2e-5
```

### Testing Classification Model

```bash
python -m core.test_classification --model_path models/classification_model.pt
```

### Testing Sentence Transformer

```bash
python -m test.test_sentence_transformer
```

### Generating Sentence Embeddings

```python
from core.sentence_transformer import SentenceTransformer
from transformers import AutoTokenizer

# Initialize model and tokenizer
model = SentenceTransformer(
    model_name="bert-base-uncased",
    embedding_dim=768,
    pooling_strategy="mean",
    normalize=True
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode sentences
sentences = ["This is a sample sentence.", "Another example text."]
encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Generate embeddings
embeddings = model(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"]
)
```

## Model Architecture

### SentenceTransformer

The SentenceTransformer class provides a flexible framework for generating sentence embeddings:

- Supports different pre-trained transformer models
- Multiple pooling strategies (mean, cls)
- Optional projection layer for custom embedding dimensions
- Normalization options for stable embeddings

### MultiTaskTransformer

The MultiTaskTransformer extends SentenceTransformer to handle specific NLP tasks:

- Classification mode: Adds a classification head for sentiment analysis
- NER mode: Provides token-level predictions for named entity recognition
- Shared backbone architecture for transfer learning opportunities

## Datasets

The framework supports:

- **IMDB Dataset**: Binary sentiment classification (positive/negative reviews)
- **CoNLL-2003 Dataset**: Named entity recognition with 9 tag classes:
  - O (Outside any entity)
  - B-PER, I-PER (Person)
  - B-ORG, I-ORG (Organization)
  - B-LOC, I-LOC (Location)
  - B-MISC, I-MISC (Miscellaneous)

## Training Details

### Classification

- Binary classification (positive/negative)
- Stratified data splitting to maintain class balance
- Cross-entropy loss function
- Early stopping based on validation loss

### Named Entity Recognition

- 9-class token classification
- Weighted loss function to handle class imbalance
- Entity-level evaluation metrics
- Support for both token-level and entity-level metrics

## Evaluation

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### NER Metrics

- Token-level accuracy
- Entity-level precision, recall, and F1 score
- Per-entity type metrics
- Macro and micro averages

## Results

### Classification Task

The sentiment classification model was trained for 3 epochs on a subset of 500, with the following results:

```
Training Progress:
Epoch 1: Train Loss: 0.6973, Val Loss: 0.5565, Accuracy: 0.6533
Epoch 2: Train Loss: 0.4356, Val Loss: 0.3445, Accuracy: 0.8267
Epoch 3: Train Loss: 0.2334, Val Loss: 0.2754, Accuracy: 0.9067

Test Metrics (Classification):
Loss: 0.3647
Accuracy: 0.8933
Precision: 0.9039
Recall: 0.8933
F1: 0.8928
```

The model showed steady improvement across all epochs with final test accuracy reaching 89.33%, suggesting successful learning of sentiment patterns in the IMDB dataset. The close alignment between precision and recall (90.39% and 89.33% respectively) indicates balanced performance across both positive and negative classes.

### Named Entity Recognition Task

The NER model was also trained on a subset of 500 samples and for 3 epochs:

```
Training Progress:
Epoch 1: Train Loss: 1.6297, Val Loss: 1.0965, Accuracy: 0.6458
Epoch 2: Train Loss: 0.8643, Val Loss: 0.7044, Accuracy: 0.6795
Epoch 3: Train Loss: 0.5671, Val Loss: 0.5222, Accuracy: 0.6895

Dataset Distribution:
Training set: 338 samples, 5040 tokens
- O (Outside entity): 82.64%
- Person entities: 5.23%
- Organization entities: 5.57%
- Location entities: 4.31%
- Miscellaneous entities: 2.25%

Test Metrics (NER):
Token Accuracy: 0.9195
Macro Precision: 6.9848
Macro Recall: 0.4539
Macro F1: 0.7503
```

Entity-specific performance varied significantly:
- Person entities (B-PER, I-PER): Strong recall (76-88%) but precision issues
- Organization entities (B-ORG): Balanced performance with F1 of 0.8235
- Location entities (B-LOC): Good recall (85%) but lower precision
- Miscellaneous entities (B-MISC, I-MISC): Poor performance, likely due to limited training examples

The high token accuracy (91.95%) compared to lower entity-level metrics illustrates a common challenge in NER tasks: the model excels at identifying non-entity tokens (the majority class "O") but struggles with consistent entity boundary detection and classification. This is particularly evident in the low precision for some entity types and zero recall for I-LOC and I-MISC categories.

The limited training data (only 500 samples) likely contributed to the inconsistent entity-level performance. Increasing the training dataset size and potentially applying more aggressive class weighting could improve results for the less common entity types.
