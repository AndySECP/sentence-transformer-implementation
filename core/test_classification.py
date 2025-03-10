"""
Testing script for the trained classification model.
Allows testing on specific examples to verify model performance.
"""

import torch
from transformers import AutoTokenizer
import argparse
import sys
import os

sys.path.append('.')

# Import the model class
from core.multi_task import MultiTaskTransformer

def load_model(model_path, model_name="bert-base-uncased", num_classes=2):
    """Load the trained classification model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}")
    model = MultiTaskTransformer(
        model_name=model_name,
        embedding_dim=768,
        num_classes=num_classes,
        num_ner_tags=9,
        dropout_prob=0.1,
        task_mode="classification"
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def predict_text(model, tokenizer, text, device):
    """Make prediction for a single text example."""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return predictions[0]

def main():
    parser = argparse.ArgumentParser(description='Test the trained classification model')
    parser.add_argument('--model_path', type=str, default='models/classification_model.pt',
                      help='Path to the saved model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                      help='Name of the base transformer model')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes for classification')
    
    args = parser.parse_args()
    
    # load the model
    model, device = load_model(args.model_path, args.model_name, args.num_classes)
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # define test examples
    test_examples = [
        {
            "text": "This movie was fantastic! I loved every minute of it.",
            "expected_label": 1,  # positive sentiment (assuming binary classification)
            "label_name": "Positive"
        },
        {
            "text": "The worst film I've ever seen. Complete waste of time and money.",
            "expected_label": 0,  # negative sentiment
            "label_name": "Negative"
        },
        {
            "text": "The acting was great but the story was a bit slow.",
            "expected_label": 1,  # positive sentiment (slightly)
            "label_name": "Positive"
        },
        {
            "text": "I fell asleep during this boring movie. The plot made no sense.",
            "expected_label": 0,  # negative sentiment
            "label_name": "Negative"
        }
    ]
    
    print("\nTesting model on specific examples:")
    print("-" * 80)
    
    for i, example in enumerate(test_examples, 1):
        text = example["text"]
        expected = example["expected_label"]
        label_name = example["label_name"]
        
        prediction = predict_text(model, tokenizer, text, device)
        
        print(f"Example {i}:")
        print(f"Text: {text}")
        print(f"Expected: {label_name} (class {expected})")
        
        pred_label = "Positive" if prediction == 1 else "Negative"
        
        print(f"Predicted: {pred_label} (class {prediction})")
        
        if prediction == expected:
            print("✓ Correct")
        else:
            print("✗ Incorrect")
        print("-" * 80)

    # compute overall accuracy on test set
    correct = sum(1 for i, ex in enumerate(test_examples) 
                 if predict_text(model, tokenizer, ex["text"], device) == ex["expected_label"])
    print(f"\nOverall accuracy on test examples: {correct}/{len(test_examples)} "
         f"({correct/len(test_examples)*100:.2f}%)")

if __name__ == "__main__":
    main()
