from typing import Optional, Literal, Union, List, Dict

import torch
import torch.nn as nn
from transformers import AutoModel
from lightning_uq_box.uq_methods.deep_evidential_regression import DERLayer

from tackai.models.regression_head import RegressionHead


class BERTBackbone(nn.Module):
    """
    BERT backbone with optional weight freezing.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze_bert: bool = False,
        freeze_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.hidden_dim = self.bert.config.hidden_size
        
        # Apply freezing
        if freeze_bert:
            self._freeze_bert_weights()
        elif freeze_layers is not None:
            self._freeze_specific_layers(freeze_layers)
    
    def _freeze_bert_weights(self):
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("Frozen all BERT weights")
    
    def _freeze_specific_layers(self, layer_indices: List[int]):
        """Freeze specific BERT layers."""
        for idx in layer_indices:
            if 0 <= idx < len(self.bert.encoder.layer):
                for param in self.bert.encoder.layer[idx].parameters():
                    param.requires_grad = False
                print(f"Frozen BERT layer {idx}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)
        
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Get a pooled representation of the hidden states
        # (batch_size, sequence_len, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(outputs.last_hidden_state, dim=1)
        # x = torch.mean(outputs.last_hidden_state, dim=1)
        # return self.dropout(self.gelu(x))

        # # Use [CLS] token representation
        # pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        # pooled_output = self.dropout(pooled_output)
        
        # return pooled_output

class BERTModel(nn.Module):
    """
    Complete BERT model with linear heads for regression.
    
    TODO: Merge with MultiEmbeddingsRegressionModel?
    """
    def __init__(
        self,
        n_targets: int = 1,
        head_type: Literal["single", "multi"] = "single",
        head_depth: Union[int, List[int]] = 1,
        head_dropout: float = 0.1,
        task_type: Literal["der", "mve", "msle"] = "der",
        freeze_bert: bool = False,
        freeze_layers: Optional[List[int]] = None,
        model_name: str = "google-bert/bert-base-cased",
        bert_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_targets = n_targets
        self.head_type = head_type
        self.task_type = task_type
        
        # BERT backbone
        self.backbone = BERTBackbone(
            model_name=model_name,
            freeze_bert=freeze_bert,
            freeze_layers=freeze_layers,
            dropout=bert_dropout,
        )
        
        # Regression head
        self.regression_head = RegressionHead(
            n_targets=n_targets,
            hidden_dim=self.backbone.hidden_dim,
            head_type=head_type,
            depth=head_depth,
            dropout=head_dropout,
            task_type=task_type,
        )
        
        # DER layer: some of the returned parameters must be positive, so the
        # layer applies a softplus to ensure positivity
        if task_type == "der":
            self.der_layer = DERLayer()
    
    def forward(self, batch: Dict[str, torch.Tensor], return_embeddings: bool = False) -> torch.Tensor:
        # Get BERT embeddings
        embeddings = self.backbone(batch)
        
        # Pass through regression head
        x = self.regression_head(embeddings)
        
        # Apply DER layer
        if self.task_type == "der":
            x = self.der_layer(x)
        
        if return_embeddings:
            return x, embeddings
        return x