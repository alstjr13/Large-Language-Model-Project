# import all necessary modules
import os
import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch

# BERT Large (cased)
# Input size: 512 Tokens (Max)
model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

# Load sample data set with incentivized reviews as pd.DataFrame
df = pd.read_csv('sample_6k_reviews_for_RA.csv')

# Split the data into training and test sets:
train_text, test_text, train_label, test_label = train_test_split()

# Tokenize input text
input_text = "This is an example sentence."
input_tokens = tokenizer(input_text, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    outputs = model(**input_tokens)
    last_hidden_states = outputs.last_hidden_state

# Print shape of embeddings
print(dataset)
