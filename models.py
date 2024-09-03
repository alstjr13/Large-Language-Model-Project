import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.num_labels = num_labels

        self._init_weight()

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.uniform_(m.bias)

        def forward(self, input_ids, attention_mask):
            pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return pooler['logits']