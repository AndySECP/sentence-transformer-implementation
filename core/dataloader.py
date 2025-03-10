"""
Task-specific data loading function that only processes and displays information
relevant to the requested task (classification or NER).
"""

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import Counter

class MultiTaskDataset(Dataset):
    """Dataset for handling both classification and NER tasks."""
    
    def __init__(self, inputs, classification_labels=None, ner_labels=None):
        self.inputs = inputs
        self.classification_labels = classification_labels
        self.ner_labels = ner_labels
    
    def __len__(self):
        return len(self.inputs["input_ids"])
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        
        if self.classification_labels is not None:
            item["classification_labels"] = self.classification_labels[idx]
        
        if self.ner_labels is not None:
            item["ner_labels"] = self.ner_labels[idx]
        
        return item

def stratified_split(dataset, labels, val_size=0.15, test_size=0.15, random_state=42):
    """
    Perform a stratified split of the dataset based on classification labels.
    
    Args:
        dataset: The full dataset to split
        labels: The classification labels for stratification
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices
    """
    # first split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # extract labels for the train+val set
    train_val_labels = [labels[i] for i in train_val_indices]
    
    # second split: train vs val
    # adjust val_size to be relative to the train+val set
    relative_val_size = val_size / (1 - test_size)
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    return train_indices, val_indices, test_indices

def load_classification_data(tokenizer, max_samples=10000, trust_remote_code=True):
    """
    Load and preprocess only classification data (IMDB).
    
    Args:
        tokenizer: The tokenizer to use for preprocessing
        max_samples: Maximum number of samples to use
        trust_remote_code: Whether to trust remote code for dataset loading
        
    Returns:
        Dictionary of classification datasets
    """
    print("Loading classification dataset (IMDB)...")
    
    # load IMDB dataset
    imdb_dataset = load_dataset("imdb", trust_remote_code=trust_remote_code)
    
    print("Preprocessing classification dataset...")
    
    # limit dataset size if needed
    if max_samples and max_samples < len(imdb_dataset["train"]):
        # ensure balanced classes when limiting
        pos_indices = [i for i, example in enumerate(imdb_dataset["train"]) 
                        if example["label"] == 1][:max_samples//2]
        neg_indices = [i for i, example in enumerate(imdb_dataset["train"]) 
                        if example["label"] == 0][:max_samples//2]
        
        # combine and shuffle indices
        selected_indices = pos_indices + neg_indices
        random.shuffle(selected_indices)
        
        # make a subset
        imdb_train = Subset(imdb_dataset["train"], selected_indices)
    else:
        imdb_train = imdb_dataset["train"]
    
    # get all IMDB data and labels
    imdb_texts = [imdb_train[i]["text"] for i in range(len(imdb_train))]
    imdb_labels = [imdb_train[i]["label"] for i in range(len(imdb_train))]
    
    imdb_inputs = tokenizer(
        imdb_texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Cconvert labels to tensor
    imdb_labels = torch.tensor(imdb_labels)
    
    # create full dataset
    imdb_full_dataset = MultiTaskDataset(imdb_inputs, classification_labels=imdb_labels)
    
    # now perform stratified split for classification data
    train_indices, val_indices, test_indices = stratified_split(
        imdb_full_dataset, 
        imdb_labels.numpy(),
        val_size=0.15, 
        test_size=0.15
    )
    
    # create stratified subsets
    imdb_train_dataset = Subset(imdb_full_dataset, train_indices)
    imdb_val_dataset = Subset(imdb_full_dataset, val_indices)
    imdb_test_dataset = Subset(imdb_full_dataset, test_indices)
    
    # verify class distribution in each split for classification
    def calculate_class_distribution(dataset_indices):
        labels = [imdb_labels[i].item() for i in dataset_indices]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        pos_percent = (pos_count / len(labels)) * 100
        neg_percent = (neg_count / len(labels)) * 100
        return {
            "total": len(labels),
            "positive": pos_count,
            "negative": neg_count,
            "positive_percent": pos_percent,
            "negative_percent": neg_percent
        }
    
    train_dist = calculate_class_distribution(train_indices)
    val_dist = calculate_class_distribution(val_indices)
    test_dist = calculate_class_distribution(test_indices)
    
    print("\nClass Distribution After Stratified Split:")
    print(f"Train: Total={train_dist['total']}, "
          f"Pos={train_dist['positive']} ({train_dist['positive_percent']:.1f}%), "
          f"Neg={train_dist['negative']} ({train_dist['negative_percent']:.1f}%)")
    print(f"Val  : Total={val_dist['total']}, "
          f"Pos={val_dist['positive']} ({val_dist['positive_percent']:.1f}%), "
          f"Neg={val_dist['negative']} ({val_dist['negative_percent']:.1f}%)")
    print(f"Test : Total={test_dist['total']}, "
          f"Pos={test_dist['positive']} ({test_dist['positive_percent']:.1f}%), "
          f"Neg={test_dist['negative']} ({test_dist['negative_percent']:.1f}%)")
    
    datasets = {
        "train": imdb_train_dataset,
        "validation": imdb_val_dataset,
        "test": imdb_test_dataset
    }
    
    print(f"\nCreated classification datasets: Train={len(train_indices)} samples")
    
    return datasets

def load_ner_data(tokenizer, max_samples=None, trust_remote_code=True):
    """
    Load and preprocess only NER data (CoNLL-2003) with optional sample limiting.
    
    Args:
        tokenizer: The tokenizer to use for preprocessing
        max_samples: Maximum number of samples to use (applies to total dataset)
        trust_remote_code: Whether to trust remote code for dataset loading
        
    Returns:
        Dictionary of NER datasets
    """
    print("Loading NER dataset (CoNLL-2003)...")
    
    # load CoNLL-2003 dataset
    conll_dataset = load_dataset("conll2003", trust_remote_code=trust_remote_code)
    
    print("Preprocessing NER dataset...")
    
    # limit the entire dataset before splitting
    if max_samples and max_samples < (len(conll_dataset["train"]) + len(conll_dataset["validation"]) + len(conll_dataset["test"])):
        print(f"Limiting NER dataset to approximately {max_samples} total samples")
        
        # calculate proportional sample sizes to maintain the original distribution
        total_original = len(conll_dataset["train"]) + len(conll_dataset["validation"]) + len(conll_dataset["test"])
        train_ratio = len(conll_dataset["train"]) / total_original
        val_ratio = len(conll_dataset["validation"]) / total_original
        test_ratio = len(conll_dataset["test"]) / total_original
        
        # calculate new sizes
        train_size = int(max_samples * train_ratio)
        val_size = int(max_samples * val_ratio) 
        test_size = max_samples - train_size - val_size
        
        train_indices = random.sample(range(len(conll_dataset["train"])), train_size)
        val_indices = random.sample(range(len(conll_dataset["validation"])), val_size)
        test_indices = random.sample(range(len(conll_dataset["test"])), test_size)
        
        limited_train = conll_dataset["train"].select(train_indices)
        limited_val = conll_dataset["validation"].select(val_indices)
        limited_test = conll_dataset["test"].select(test_indices)
    else:
        limited_train = conll_dataset["train"]
        limited_val = conll_dataset["validation"]
        limited_test = conll_dataset["test"]
        if max_samples:
            print(f"Requested sample limit ({max_samples}) exceeds available samples ({len(conll_dataset['train']) + len(conll_dataset['validation']) + len(conll_dataset['test'])})")
    
    def analyze_ner_tags(dataset, split_name):
        tag_counts = Counter()
        total_tokens = 0
        
        for example in dataset:
            tags = example["ner_tags"]
            tag_counts.update(tags)
            total_tokens += len(tags)
        
        # map tag IDs to names (CoNLL specific)
        tag_names = {
            0: "O",  # Outside any entity
            1: "B-PER", 2: "I-PER",  # Person
            3: "B-ORG", 4: "I-ORG",  # Organization
            5: "B-LOC", 6: "I-LOC",  # Location
            7: "B-MISC", 8: "I-MISC"  # Miscellaneous
        }
        
        print(f"\n{split_name} set:")
        print(f"Total samples: {len(dataset)}")
        print(f"Total tokens: {total_tokens}")
        print("Entity distribution:")
        for tag_id, count in sorted(tag_counts.items()):
            tag_name = tag_names.get(tag_id, f"Unknown-{tag_id}")
            percentage = (count / total_tokens) * 100
            print(f"  {tag_name}: {count} ({percentage:.2f}%)")
    
    print("\nNER Dataset Statistics:")
    analyze_ner_tags(limited_train, "Training")
    analyze_ner_tags(limited_val, "Validation")
    analyze_ner_tags(limited_test, "Test")
    
    def tokenize_and_align_labels(examples):
        """Tokenize and align labels for NER task."""
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, 
            padding="max_length", max_length=512
        )
        
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        return tokenized_inputs, labels
    
    train_inputs, train_labels = tokenize_and_align_labels(limited_train)
    val_inputs, val_labels = tokenize_and_align_labels(limited_val)
    test_inputs, test_labels = tokenize_and_align_labels(limited_test)
    
    # convert to PyTorch tensors
    def convert_to_tensors(inputs, labels):
        tensor_inputs = {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
        }
        tensor_labels = torch.tensor(labels)
        return tensor_inputs, tensor_labels
    
    train_tensor_inputs, train_tensor_labels = convert_to_tensors(train_inputs, train_labels)
    val_tensor_inputs, val_tensor_labels = convert_to_tensors(val_inputs, val_labels)
    test_tensor_inputs, test_tensor_labels = convert_to_tensors(test_inputs, test_labels)
    
    # create NER datasets
    ner_train_dataset = MultiTaskDataset(train_tensor_inputs, ner_labels=train_tensor_labels)
    ner_val_dataset = MultiTaskDataset(val_tensor_inputs, ner_labels=val_tensor_labels)
    ner_test_dataset = MultiTaskDataset(test_tensor_inputs, ner_labels=test_tensor_labels)
    
    datasets = {
        "train": ner_train_dataset,
        "validation": ner_val_dataset,
        "test": ner_test_dataset
    }
    
    print(f"\nCreated NER datasets: Train={len(ner_train_dataset)}, "
          f"Validation={len(ner_val_dataset)}, Test={len(ner_test_dataset)} samples")
    
    return datasets

def load_and_preprocess_data(tokenizer, max_imdb_samples=10000, max_ner_samples=None, trust_remote_code=True, task="both"):
    """
    Load and preprocess data based on the specified task.
    
    Args:
        tokenizer: The tokenizer to use for preprocessing
        max_imdb_samples: Maximum number of IMDB samples to use
        max_ner_samples: Maximum number of NER samples to use (if None, uses max_imdb_samples)
        trust_remote_code: Whether to trust remote code for dataset loading
        task: Which task to load data for ("classification", "ner", or "both")
        
    Returns:
        Dictionary of datasets for the specified task(s)
    """
    # if max_ner_samples not specified, use the same value as max_imdb_samples
    if max_ner_samples is None:
        max_ner_samples = max_imdb_samples
    
    if task == "classification" or task == "both":
        classification_datasets = load_classification_data(tokenizer, max_imdb_samples, trust_remote_code)
    
    if task == "ner" or task == "both":
        ner_datasets = load_ner_data(tokenizer, max_ner_samples, trust_remote_code)
    
    # prepare the final datasets dictionary
    if task == "classification":
        return classification_datasets
    elif task == "ner":
        return ner_datasets
    else:
        combined_datasets = {}
        
        # add classification datasets
        for split, dataset in classification_datasets.items():
            combined_datasets[split] = dataset
        
        # add NER datasets with "ner_" prefix
        for split, dataset in ner_datasets.items():
            combined_datasets[f"ner_{split}"] = dataset
        
        return combined_datasets

if __name__ == "__main__":
    # just for testing the function    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # test with different task parameters
    for task in ["classification", "ner"]:
        print(f"\n\n===== Testing {task.upper()} task =====")
        datasets = load_and_preprocess_data(tokenizer, max_imdb_samples=1000, task=task)
        
        print(f"\nDatasets loaded for {task}:")
        for split, dataset in datasets.items():
            print(f"  {split}: {len(dataset)} samples")
