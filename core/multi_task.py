import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .sentence_transformer import SentenceTransformer

class MultiTaskTransformer(SentenceTransformer):
    """Multi-task transformer for classification or NER."""
    
    def __init__(
        self,
        model_name="bert-base-uncased",
        embedding_dim=768,
        num_classes=3,  # for classification
        num_ner_tags=9,  # for NER
        dropout_prob=0.1,
        task_mode="classification"  # "classification" or "ner"
    ):
        super().__init__(model_name, embedding_dim)
        
        self.task_mode = task_mode
        
        if task_mode == "classification":
            # classification head
            self.classification_head = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(embedding_dim // 2),
                nn.Linear(embedding_dim // 2, num_classes)
            )
            self.classification_norm = nn.LayerNorm(embedding_dim)
        
        elif task_mode == "ner":
            # NER head
            self.ner_head = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(embedding_dim // 2),
                nn.Linear(embedding_dim // 2, num_ner_tags)
            )
            self.ner_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids, attention_mask):
        """Forward pass for the selected task."""
        if self.task_mode == "classification":
            # get sentence embeddings from parent class
            embeddings = super().forward(input_ids, attention_mask)
            return self.classification_head(self.classification_norm(embeddings))
        
        elif self.task_mode == "ner":
            # for NER, we need token-level predictions
            token_embeddings = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            
            return self.ner_head(self.ner_norm(token_embeddings))
