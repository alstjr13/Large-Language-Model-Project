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
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_score, \
    recall_score, f1_score
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

#model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
model_name = 'bert-large-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-cased-whole-word-masking-finetuned-squad")
#model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-cased-whole-word-masking-finetuned-squad")

class BertReview:

    def __init__(self, df):
        self.df = df
        print("Initializing BERT Large (Cased)")
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained('bert-large-cased',
                                                                   num_labels= 2,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)
        #self.model.classifier.weight = torch.nn.Parameter(torch.randn(self.model.classifier.weight.size()))
        # Initialize bias
        #self.model.classifier.bias = torch.nn.Parameter(torch.zeros(self.model.classifier.bias.size()))
        #self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.cuda()

    def splitData(self):
        print("Spliting Data")
        self.incentivizedReview = self.df[self.df['incentivized_999'] == 1]
        self.notIncentivizedReview = self.df[self.df['incentivized_999'] == 0]

    def tokenizeData(self, texts):
        return self.tokenizer(texts, truncation=True, return_tensors="pt")

    def create_dataloader(self, texts, labels, batch_size = 8):
        tokenized_data = self.tokenizeData(texts)
        dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention mask'], torch.tensor(labels))
        return DataLoader(dataset, batch_size= batch_size, shuffle= True)

    def train(self, epochs = 3, batch_size = 8):
        self.splitData()

        texts = self.df['cleanedText'].tolist()
        labels = self.df['incentivized_999'].tolist()

        dataloader = self.create_dataloader(texts, labels, batch_size)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        loss = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask)
                loss = loss(output.logits, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self):
        self.model.eval()
        texts = self.df['cleanedText'].tolist()
        labels = self.df['incentivized_999'].tolist()

        dataloader = self.create_dataloader(texts, labels, batch_size=8)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)

        return accuracy, precision, recall, f1, auc

    def plot_metrics(self, metrics):
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        plt.figure(figsize=(10, 5))
        plt.bar(labels, metrics)
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.ylim(0, 1)
        plt.show()



processor = BertReview(df)
















"""
class BertIncentivized(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length = 512):
        self.texts = texts,
        self.labels = labels,
        self.tokenizer = tokenizer,
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx],
        label = self.labels[idx],
        encodings = self.tokenizer.encode_plus(
            text,
            max_length = self.max_length,
            return_tensors = "pt",
            truncation = False
        )
        return {
            'texts': text,
            "input_ids": encodings['input_ids'].flatten(),
            "attention_mask": encodings['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = ReviewDataset(train_text.tolist(), train_label.tolist(), tokenizer)
test_dataset = ReviewDataset(test_text.tolist(), train_label.tolist(), tokenizer)
"""

