# TODO
# Make 5 figures of epoch = 1,2,3 and test of the following criterias:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1 Score
# 5. AUC
import numpy as np
import pandas as pd
import re
from collections import defaultdict

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

import matplotlib.pyplot as plt

import torch

# Load sample data set with incentivized reviews as pd.DataFrame
df = pd.read_csv('cleaned_reviews_with_labels.csv')

# Split the data into training and test sets:
train_text, test_text, train_label, test_label = train_test_split(df['cleanedReviewText'],
                                                                  df['incentivized_999'],
                                                                  test_size=0.2,
                                                                  random_state=42)

# BERT Large (cased)
# Input size: 512 Tokens (Max)
model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        texts=df.cleanedReviewText.to_numpy(),
        labels=df.incentivized_999.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

BATCH_SIZE = 8
MAX_LEN = 512

train_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 3

def train_epoch(
  model,
  data_loader,
  optimizer,
  device,
  scheduler,
  n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    correct_predictions = 0
    losses = []
    predictions = []
    labels = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels_batch = d["label"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels_batch)
            losses.append(loss.item())
            predictions.extend(preds)
            labels.extend(labels_batch)
    predictions = torch.stack(predictions).cpu()
    labels = torch.stack(labels).cpu()
    return correct_predictions.double() / n_examples, np.mean(losses), predictions, labels

def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auc = roc_auc_score(labels, preds)
    return accuracy, precision, recall, f1, auc

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler=None,
        n_examples=len(train_text)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss, preds, labels = eval_model(
        model,
        test_data_loader,
        device,
        n_examples=len(test_text)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    val_acc, val_precision, val_recall, val_f1, val_auc = compute_metrics(preds, labels)
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    history['val_f1'].append(val_f1)
    history['val_auc'].append(val_auc)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# Plot the metrics
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(18, 6))

plt.subplot(1, 5, 1)
plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 5, 2)
plt.plot(epochs, history['val_precision'], label='Validation Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 5, 3)
plt.plot(epochs, history['val_recall'], label='Validation Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 5, 4)
plt.plot(epochs, history['val_f1'], label='Validation F1 Score')
plt.title('F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 5, 5)
plt.plot(epochs, history['val_auc'], label='Validation AUC')
plt.title('AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.show()