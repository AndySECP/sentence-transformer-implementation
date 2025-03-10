import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Optional
import random

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

# Simple task-specific loss functions
class ClassificationLoss(nn.CrossEntropyLoss):
    """Classification loss wrapper."""
    pass

class WeightedNERLoss(nn.Module):
    """
    NER loss function that applies higher weights to non-O entity tokens.
    This encourages the model to focus on detecting named entities rather
    than just predicting the majority 'O' class.
    """
    def __init__(self, id2tag, non_o_weight=5.0, ignore_index=-100):
        """
        Args:
            id2tag: Dictionary mapping tag IDs to tag names
            non_o_weight: Weight multiplier for non-O entity tokens
            ignore_index: Index to ignore (padding)
        """
        super().__init__()
        self.ignore_index = ignore_index
        
        # find the number of classes (excluding padding)
        num_classes = max([tag_id for tag_id in id2tag.keys() if tag_id != ignore_index]) + 1
        
        # create class weights: higher weights for non-O tokens
        self.class_weights = torch.ones(num_classes)
        for tag_id, tag_name in id2tag.items():
            if tag_id != 0 and tag_id != ignore_index and tag_id < num_classes:  # skip 'O' and padding
                self.class_weights[tag_id] = non_o_weight
    
    def forward(self, logits, labels, attention_mask):
        batch_size, seq_len, num_tags = logits.shape
        logits_flat = logits.view(-1, num_tags)
        labels_flat = labels.view(-1)
        
        # only consider non-padding tokens
        active_mask = (labels_flat != self.ignore_index)
        active_logits = logits_flat[active_mask]
        active_labels = labels_flat[active_mask]
        
        weights = self.class_weights.to(logits.device)
        
        # use CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean', ignore_index=self.ignore_index)
        loss = criterion(active_logits, active_labels)
        
        return loss

def train_epoch(model, dataloader, optimizer, loss_fn, device, is_ner=False):
    """Generic training for one epoch."""
    model.train()
    epoch_loss = 0.0
    steps = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        if is_ner:
            labels = batch['ner_labels'].to(device)
        else:
            labels = batch['classification_labels'].to(device)
        
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # calculate loss
        if is_ner:
            loss = loss_fn(outputs, labels, attention_mask)
        else:
            loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        steps += 1
    
    return epoch_loss / steps

def evaluate(model, dataloader, loss_fn, device, is_ner=False):
    """Generic evaluation function."""
    model.eval()
    total_loss = 0.0
    steps = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if is_ner:
                labels = batch['ner_labels'].to(device)
            else:
                labels = batch['classification_labels'].to(device)
            
            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # calculate loss
            if is_ner:
                loss = loss_fn(outputs, labels, attention_mask)
                # token predictions
                preds = torch.argmax(outputs, dim=2)
                
                # only consider non-padding tokens for accuracy
                for i in range(len(preds)):
                    for j in range(len(preds[i])):
                        if attention_mask[i][j] == 1:
                            all_preds.append(preds[i][j].item())
                            all_labels.append(labels[i][j].item())
            else:
                loss = loss_fn(outputs, labels)
                # get sentence predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            steps += 1
    
    metrics = {
        'loss': total_loss / steps,
        'accuracy': accuracy_score(all_labels, all_preds)
    }
    
    # add more detailed metrics for classification (precision/recall/f1)
    if not is_ner:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return metrics

def train_model(
    model, 
    dataloaders, 
    optimizer, 
    scheduler, 
    device, 
    save_path, 
    is_ner=False,
    num_epochs=5, 
    patience=3
):
    """Train either classification or NER model."""
    # initialize loss function
    loss_fn = WeightedNERLoss(id2tag, non_o_weight=5.0) if is_ner else ClassificationLoss()
    
    # training variables
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # do the training
        train_loss = train_epoch(
            model, dataloaders['train'], optimizer, loss_fn, device, is_ner)
        
        # evaluate
        val_metrics = evaluate(
            model, dataloaders['val'], loss_fn, device, is_ner)
        
        # update learning rate if needed
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['accuracy'].append(val_metrics['accuracy'])
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # check if we have the best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
        
        # early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"Best model was from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    return history


def evaluate_ner(model, dataloader, device, id2tag):
    """
    Evaluate NER model with entity-specific metrics.
    
    Args:
        model: The trained model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        id2tag: Mapping from tag IDs to tag names (e.g., {1: "B-PER", 2: "I-PER", ...})
    
    Returns:
        Dictionary with detailed evaluation metrics
    """
    
    model.eval()
    
    # For token-level metrics
    all_true_ids = []
    all_pred_ids = []
    
    # For seqeval-style entity-level evaluation (if needed later)
    all_true_tag_sequences = []
    all_pred_tag_sequences = []
    
    # we will track per-class metrics
    class_correct = {}
    class_total = {}
    for tag_id in id2tag:
        if tag_id != -100:  # skip padding index
            class_correct[tag_id] = 0
            class_total[tag_id] = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["ner_labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs  # no need to access .logits since outputs is the logits directly
            
            # get predictions
            predictions = torch.argmax(logits, dim=2)
            
            # only evaluate on actual tokens (not padding)
            active_mask = labels != -100
            
            # calculate per-class metrics
            for tag_id in class_total.keys():
                tag_mask = (labels == tag_id) & active_mask
                class_total[tag_id] += tag_mask.sum().item()
                class_correct[tag_id] += ((predictions == tag_id) & tag_mask).sum().item()
            
            # collect token IDs for sklearn metrics
            for i in range(input_ids.size(0)):  # we go through sequences in the batch
                true_seq = []
                pred_seq = []
                active_seq_mask = active_mask[i]
                
                for j in range(active_seq_mask.sum().item()):
                    idx = j
                    true_id = labels[i][active_seq_mask][idx].item()
                    pred_id = predictions[i][active_seq_mask][idx].item()
                    
                    # Add to flat lists for token-level metrics
                    all_true_ids.append(true_id)
                    all_pred_ids.append(pred_id)
                    
                    # convert ids to tag names (for sequence-level evaluation if needed later)
                    true_tag = id2tag.get(true_id, "O")
                    pred_tag = id2tag.get(pred_id, "O")
                    
                    true_seq.append(true_tag)
                    pred_seq.append(pred_tag)
                
                if true_seq:
                    all_true_tag_sequences.append(true_seq)
                    all_pred_tag_sequences.append(pred_seq)
    
    # calculate per-class precision and recall
    per_class_metrics = {}
    non_o_f1_values = []
    non_o_precision_values = []
    non_o_recall_values = []

    for tag_id, tag_name in id2tag.items():
        if tag_id == 0 or tag_id == -100:  # we skip O tag and padding
            continue
            
        if class_total[tag_id] > 0:
            precision = class_correct[tag_id] / max(sum((predictions == tag_id).cpu().numpy().flatten()), 1)
            recall = class_correct[tag_id] / max(class_total[tag_id], 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            
            per_class_metrics[tag_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': class_total[tag_id]
            }
            
            non_o_precision_values.append(precision)
            non_o_recall_values.append(recall)
            non_o_f1_values.append(f1)
    
    # Using token IDs for scikit-learn functions (not tag sequences)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # For micro metrics, we can use the flattened ID lists
    overall_metrics = {
        'micro_precision': precision_score(all_true_ids, all_pred_ids, average='micro', labels=list(id2tag.keys())[:-1], zero_division=0),
        'micro_recall': recall_score(all_true_ids, all_pred_ids, average='micro', labels=list(id2tag.keys())[:-1], zero_division=0),
        'micro_f1': f1_score(all_true_ids, all_pred_ids, average='micro', labels=list(id2tag.keys())[:-1], zero_division=0),
        'macro_precision': np.mean(non_o_precision_values) if non_o_precision_values else 0,
        'macro_recall': np.mean(non_o_recall_values) if non_o_recall_values else 0,
        'macro_f1': np.mean(non_o_f1_values) if non_o_f1_values else 0,
    }
    
    # calculate regular token-level accuracy for comparison
    total_correct = sum(class_correct.values()) 
    total_tokens = sum(class_total.values())
    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    # Instead of using sequences for the classification report, use the token IDs
    report = classification_report(
        all_true_ids, 
        all_pred_ids, 
        labels=list(id2tag.keys())[:-1],  # Exclude padding token
        target_names=[id2tag[i] for i in id2tag if i != -100],  # Names corresponding to the labels
        output_dict=True,
        zero_division=0
    )
    
    return {
        'token_accuracy': token_accuracy,
        'overall': overall_metrics,
        'per_class': per_class_metrics,
        'full_report': report
    }
