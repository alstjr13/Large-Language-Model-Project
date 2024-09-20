import os
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Create a ReviewDataset with inputs:
# texts: reviews of the users (either incentivized or unincentivized - labelled with 0 or 1)
# labels: corresponding label value --> 0 or 1
# tokenizer: BERT-Large Tokenizer
# max_length: set default to 512
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            #'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Resulting data split:
# Train = 90 %
# Test = 10 %
def data_split(X, y):
    """
    Split data into 3 different dataset
    80% Train dataset
    20% Test dataset
    :param X: pd.DataFrame including review texts, string
    :param y: pd.DataFrame including labels, either 0 or 1
    :return: texts (train), labels (train), texts (test), labels (test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9, test_size=0.1, random_state=42, shuffle=True)

    return X_train, y_train, X_test, y_test

def is_nan(x) -> bool:
    """
    Checks whether if the dataset includes a NaN value
    :param x: dataset to feed into the custom dataset
    :return: single bool, True -> x contains NaN, False -> x does not contain NaN
    """
    return x.isnull().values.any()

