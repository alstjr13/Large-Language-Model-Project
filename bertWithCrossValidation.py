import os
import warnings
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings('ignore')

# Load sample dataset:
# reviewText                     (column 0): reviews in text... including incentivized and non-incentivized reviews
# incentivized_999               (column 1): - 0 : non-incentivized reviews
#                                            - 1 : incentivized reviews
# incent_bert_highest_score_sent (column 2): disclosure sentence
filePath = '../data/updated_review_sample_for_RA.csv'
df = pd.read_csv(filePath)

# CHECK if any row has NaN value
print(df['reviewText'].isna().any())        # True
print(df['incentivized_999'].isna().any())  # False

# Delete any row that has NaN value
df = df.dropna(subset=['reviewText'])

# Take random samples from the dataset (.csv file)
notIncentivized = df[df['incentivized_999'] == 0].sample(n=10, random_state=42)
incentivized = df[df['incentivized_999'] == 1].sample(n=10, random_state=42)

# Merge two dataFrames
# Drop column 2: 'incent_bert_highest_score_sent
newdf = pd.concat([notIncentivized, incentivized])
newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)
newdf = newdf.drop(['incent_bert_highest_score_sent'], axis=1)

X = newdf['reviewText']
y = newdf['incentivized_999']

# Split the Data into Training and Test ---> 8 : 2 Ratio
# BASED ON THE PAPER : Training   : 72%
#                      Validation : 18%
#                      Test       : 10%
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# Train, Validation split
train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=True, shuffle=True)

# Print Train, Validation, Test Set Distribution:
print(f" Training Label Distribution \n {pd.Series(train_labels).value_counts()}")
print(f" Validation Label Distribution \n {pd.Series(validation_labels).value_counts()}")
print(f" Test Label Distribution \n {pd.Series(test_labels).value_counts()}")

# Load BERT model and BERT tokenizer
model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
max_length = 512

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
validation_encodings = tokenizer(validation_texts, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create 3 different datasets: Train (72%), Validation (18%), Test (10%)
train_dataset = ReviewDataset(train_encodings, train_labels)
validation_dataset = ReviewDataset(validation_encodings, validation_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir="../results/bertWithCrossValidation",
    num_train_epochs = 4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../logs/bertWithCrossValidation',
    eval_strategy='epoch',
    save_strategy="epoch",
    logging_steps=5,
    load_best_model_at_end=True,
)

# For each epoch:
# 5 Cross validation

num_epochs = 4
for epoch in range(num_epochs):
    pass