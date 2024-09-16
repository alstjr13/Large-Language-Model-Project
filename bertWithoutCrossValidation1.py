import os
import warnings
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, \
    precision_score, recall_score, f1_score
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

import utils

# Enable memory use
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings('ignore')

# ---------------------------------------PRE-PROCESSING-DATA---------------------------------------------------------

# Load dataset:
# reviewText                     (column 0): reviews in text... including incentivized and non-incentivized reviews
# incentivized_999               (column 1): - 0 : non-incentivized reviews
#                                            - 1 : incentivized reviews
# incent_bert_highest_score_sent (column 2): sentence with highest probability of being "disclosure sentence" in reviewText
filePath = "../data/updated_review_sample_for_RA.csv"
df = pd.read_csv(filePath)

# Delete any row that has NaN value
df = df.dropna(subset=["reviewText"])

# Take random samples from the dataset
notIncentivized = df[df['incentivized_999'] == 0].sample(n=300, random_state=42)
incentivized =df[df['incentivized_999'] == 1].sample(n=300, random_state=42)

# Combine random samples
df = pd.concat([notIncentivized, incentivized])

# Drop unnecessary column
df = df.drop(['incent_bert_highest_score_sent'], axis=1)

# Reset index and shuffle sample
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.rename(columns={"reviewText" : "texts", "incentivized_999": "labels"})
X = df["texts"]
y = df["labels"]

# Split data to Train, Validation and Test (0.72 : 0.18 : 0.1 Ratio)
train_texts, train_labels, validation_texts, validation_labels, test_texts, test_labels = utils.data_split(X, y)

# Print number of labels in splited data
print(f"Training Set Distribution: \n {pd.Series(train_labels).value_counts()}")
print(f"Validation Set Distribution: \n {pd.Series(validation_labels).value_counts()}")
print(f"Test Set Distribution: \n {pd.Series(test_labels).value_counts()}")

# Initialize BERT Large Model and BERT Large Tokenizer
model = BertForSequenceClassification.from_pretrained("bert-large-cased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
max_length = 512

# Create ReviewDataset(Dataset), with encodings
trainDataset = utils.ReviewDataset(train_texts.tolist(), train_labels.tolist(), tokenizer=tokenizer, max_length=max_length)
validationDataset = utils.ReviewDataset(validation_texts.tolist(), validation_labels.tolist(), tokenizer=tokenizer, max_length=max_length)
testDataset = utils.ReviewDataset(test_texts.tolist(), test_labels.tolist(), tokenizer=tokenizer, max_length=max_length)



"""

print(trainDataset)

if __name__ == "__main__":
    pass

"""

