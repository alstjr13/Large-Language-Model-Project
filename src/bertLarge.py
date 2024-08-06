# TODO
# Make 5 figures of epoch = 1,2,3 and test of the following criterias:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1 Score
# 5. AUC

import pandas as pd
import re

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch



# BERT Large (cased)
# Input size: 512 Tokens (Max)
#model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
#tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

# Load sample data set with incentivized reviews as pd.DataFrame
df = pd.read_csv('sample_6k_reviews_for_RA.csv')

# Distinguish each columns
reviewText = df['reviewText']
incentivized_label = df['incentivized_999']

# Function to remove the sentence with ("free" or "discount" or "reduced") in df["reviewText"]
def remove_incentivized_sentence(text):
    # Add more patterns if needed
    patterns = [
        r'[^\*.(,!?]*\b(free product|free|discount|discounted rate|reduced|reduced price|Disclaimer|\*)\b[^.,!)?]*[.!?]'
    ]
    for pattern in patterns:
        if isinstance(text, str):
            return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        else:
            return ''

# Create another column "cleanedReviewText" with incentivized sentence removed
df['cleanedReviewText'] = df['reviewText'].apply(remove_incentivized_sentence)

# Create an another .csv file: cleanedReviewWithLabels.csv
cleaned_review_df = df[['cleanedReviewText', 'incentivized_999']]
cleaned_review_df.to_csv('./cleaned_reviews_with_labels.csv', index=False)

# Split the data into training and test sets:
train_text, test_text, train_label, test_label = train_test_split(cleaned_review_df['cleanedReviewText'],
                                                                  cleaned_review_df['incentivized_999'],
                                                                  test_size=0.2,
                                                                  random_state=42)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts,
        self.labels = labels,
        self.tokenizer = tokenizer,
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)


class ReviewClassifier:
    def __init__(self, model, num_labels, optimizer):
        self.model = model,
        self.num_labels = num_labels
        self.optimizer = optimizer,


# Tokenize input text
#input_text = "This is an example sentence."
#input_tokens = tokenizer(input_text, return_tensors='pt')

# Get BERT embeddings
#with torch.no_grad():
#    outputs = model(**input_tokens)
#    last_hidden_states = outputs.last_hidden_state

