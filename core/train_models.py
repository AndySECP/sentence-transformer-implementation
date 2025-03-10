#!/usr/bin/env python
"""
Training script for MultiTaskTransformer model.
Supports training for classification and NER tasks separately.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
from core.multi_task import MultiTaskTransformer
from core.dataloader import MultiTaskDataset, load_and_preprocess_data
from core.train import train_model, evaluate, evaluate_ner

id2tag = {
    0: "O",      # Outside any entity
    1: "B-PER",  # Beginning of person
    2: "I-PER",  # Inside of person
    3: "B-ORG",  # Beginning of organization
    4: "I-ORG",  # Inside of organization
    5: "B-LOC",  # Beginning of location
    6: "I-LOC",  # Inside of location
    7: "B-MISC", # Beginning of miscellaneous
    8: "I-MISC", # Inside of miscellaneous
    -100: "PAD"  # Padding token
}

def train_classification(args):
    """Train the classification task"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = MultiTaskTransformer(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        num_ner_tags=args.num_ner_tags,
        dropout_prob=args.dropout,
        task_mode="classification"
    )
    model.to(device)
    
    print("Loading and preparing datasets...")
    datasets = load_and_preprocess_data(
        tokenizer=tokenizer, 
        max_imdb_samples=args.max_samples,
        trust_remote_code=True
    )
    
    # create dataloaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'], 
            batch_size=args.batch_size, 
            shuffle=True
        ),
        'val': DataLoader(
            datasets['validation'], 
            batch_size=args.batch_size
        ),
        'test': DataLoader(
            datasets['test'], 
            batch_size=args.batch_size
        )
    }
    
    # initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    print("Starting classification training...")
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=os.path.join(args.output_dir, 'classification_model.pt'),
        is_ner=False,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # load the best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'classification_model.pt')))
    
    # evaluate on test set
    print("Evaluating classification model on test set...")
    from core.train import ClassificationLoss
    loss_fn = ClassificationLoss()
    
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders['test'],
        loss_fn=loss_fn,
        device=device,
        is_ner=False
    )
    
    print("\nTest Metrics (Classification):")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nClassification training complete!")
    return model


def train_ner(args):
    """Train the NER task"""
    print(f"DEBUG: Using max_ner_samples = {args.max_ner_samples} (max_samples = {args.max_samples})")
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = MultiTaskTransformer(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        num_ner_tags=args.num_ner_tags,
        dropout_prob=args.dropout,
        task_mode="ner"
    )
    model.to(device)
    
    print("Loading and preparing datasets...")
    datasets = load_and_preprocess_data(
        tokenizer=tokenizer, 
        max_imdb_samples=args.max_samples,
        max_ner_samples=args.max_ner_samples if args.max_ner_samples is not None else args.max_samples,
        trust_remote_code=True,
        task="ner"
    )
    
    # create dataloaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=args.batch_size, 
            shuffle=True
        ),
        'val': DataLoader(
            datasets['validation'],
            batch_size=args.batch_size
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=args.batch_size
        )
    }
    
    # initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    print("Starting NER training...")
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=os.path.join(args.output_dir, 'ner_model.pt'),
        is_ner=True,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # load the best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'ner_model.pt')))
    
    # evaluate on test set
    print("Evaluating NER model on test set...")
    from core.train import WeightedNERLoss
    loss_fn = WeightedNERLoss(id2tag, non_o_weight=5.0)
    
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders['test'],
        loss_fn=loss_fn,
        device=device,
        is_ner=True
    )
    
    print("\nTest Metrics (NER):")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nNER training complete!")
    print("Evaluating NER model on test set...")
    test_metrics = evaluate_ner(
        model=model,
        dataloader=dataloaders['test'],
        device=device,
        id2tag=id2tag
    )

    print("\nTest Metrics (NER):")
    print(f"Token Accuracy: {test_metrics['token_accuracy']:.4f}")
    print("\nEntity-Level Metrics (excluding 'O'):")
    print(f"Macro Precision: {test_metrics['overall']['macro_precision']:.4f}")
    print(f"Macro Recall: {test_metrics['overall']['macro_recall']:.4f}")
    print(f"Macro F1: {test_metrics['overall']['macro_f1']:.4f}")

    print("\nPer-Entity Type Metrics:")
    for entity, metrics in test_metrics['per_class'].items():
        print(f"{entity}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train MultiTaskTransformer model')
    
    # task selection
    parser.add_argument('--task', type=str, choices=['classification', 'ner'], 
                      required=True, help='Task to train: classification or ner')
    
    # model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='Transformer model to use')
    parser.add_argument('--embedding_dim', type=int, default=768,
                      help='Embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes for classification task')
    parser.add_argument('--num_ner_tags', type=int, default=9,
                      help='Number of NER tags')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout probability')
    
    # training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                      help='Patience for early stopping')
    parser.add_argument('--max_samples', type=int, default=10000,
                      help='Maximum number of samples to use for training (applies to both tasks)')
    parser.add_argument('--max_ner_samples', type=int, default=None,
                      help='Maximum number of NER samples to use (if not set, uses max_samples)')
    
    # hardware/output parameters
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA even if available')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save models')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # train the selected task
    if args.task == 'classification':
        print("=== Training Classification Task ===")
        classification_model = train_classification(args)
    elif args.task == 'ner':
        print("=== Training NER Task ===")
        ner_model = train_ner(args)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
