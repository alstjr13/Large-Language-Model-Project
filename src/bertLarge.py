# import all necessary modules
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
import csv
import pandas as pd

# Load BERT Large (cased)
# Input size: 512 Tokens (Max)
model = AutoModel.from_pretrained("bert-large-cased")
tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')

# No tokenization necessary in this case --> input is the sentence

dataset = pd.read_csv("sample_6k_reviews_for_RA.csv")

# Tokenize input text
input_text = "This is an example sentence."
input_tokens = tokenizer(input_text, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    outputs = model(**input_tokens)
    last_hidden_states = outputs.last_hidden_state

# Print shape of embeddings
print(dataset)
