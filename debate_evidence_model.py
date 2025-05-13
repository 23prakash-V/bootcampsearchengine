import torch
import torch.nn as nn
from transformers import DistilBertModel

class DebateEvidenceModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits 