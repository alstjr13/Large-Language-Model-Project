# TODO
# Make 5 figures of epoch = 1,2,3 and test of the following criterias:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1 Score
# 5. AUC

import numpy as np
import pandas as pd
from collections import defaultdict
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
#from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt



# Load sample data set with incentivized reviews as pd.DataFrame
df = pd.read_csv('../data/cleaned_reviews_with_labels.csv')

# Split the data into training and test sets:
train_text, test_text, train_label, test_label = train_test_split(df['cleanedReviewText'],
                                                                  df['incentivized_999'],
                                                                  test_size=0.2,
                                                                  random_state=42)
# BERT Large (cased)
# Input size: 512 Tokens (Max)
#pipe = pipeline("question-answering", model="google-bert/bert-large-cased-whole-word-masking-finetuned-squad")

model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-cased-whole-word-masking-finetuned-squad")
#model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-cased-whole-word-masking-finetuned-squad")

