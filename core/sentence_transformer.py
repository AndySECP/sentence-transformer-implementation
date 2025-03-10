import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        embedding_dim=768,
        pooling_strategy="mean",
        normalize=True
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize

        # additional projection layer to get desired embedding dimension
        if embedding_dim != self.transformer.config.hidden_size:
            self.projection = nn.Linear(
                self.transformer.config.hidden_size,
                embedding_dim
            )
        else:
            self.projection = nn.Identity()

        # layer normalization for more stable embeddings
        self.layer_norm = nn.LayerNorm(embedding_dim) if normalize else nn.Identity()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if self.pooling_strategy == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            # mean pooling with attention mask
            pooled = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(-1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # project to desired dimension and normalize if specified
        embeddings = self.projection(pooled)
        embeddings = self.layer_norm(embeddings)

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
