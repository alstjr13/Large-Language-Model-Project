import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt


# Load sample data set with incentivized reviews as pd.DataFrame
df = pd.read_csv('../data/cleaned_reviews_with_labels.csv')

# Randomly select 100 incentivized reviews (Labelled with 1)
#                 100 not incentivized reviews (Labelled with 0)
notIncentivized = df[df["incentivized_999"] == 0].sample(n=100, random_state=42)
incentivized = df[df["incentivized_999"] == 1].sample(n=100, random_state=42)

hasNaText = incentivized['cleanedReviewText'].isna().any()
hasNaLabel = incentivized['incentivized_999'].isna().any()
print(hasNaText)
print(hasNaLabel)

hasNaText = notIncentivized['cleanedReviewText'].isna().any()
hasNaLabel = notIncentivized['incentivized_999'].isna().any()
print(hasNaText)
print(hasNaLabel)

newdf = pd.concat([notIncentivized, incentivized])
newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)

#print(newdf)
# Split the data
X = newdf["cleanedReviewText"]
y = newdf["incentivized_999"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create a ReviewDataset with inputs:
# 1. texts:  reviews of the users (either incentivized or unincentivized - labelled with 0 or 1)
# 2. labels: corresponding label value --> 0 or 1
# 3. tokenizer: BERT-Large Tokenizer
# 4. max_length: set default to 4096
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=4096):
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize Bigbird-RoBERTa-large Tokenizer with maxlength =
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-large')
max_length = 4096

# Datasets to feed into BERT-Large
train_dataset = ReviewsDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length)
test_dataset = ReviewsDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length)

# Initialize BERT-large
model = AutoModelForSequenceClassification.from_pretrained('google/bigbird-roberta-large', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results/longformer',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs/longformer',
    logging_steps=10,
    eval_strategy="epoch"
)
def compute_metrics(p):
    pred = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, pred, average='binary')
    precision = precision_score(p.label_ids, pred, average='binary')
    recall = recall_score(p.label_ids, pred, average='binary')
    f1 = f1_score(p.label_ids, pred, average='binary')
    roc_auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    #precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, pred, average='binary')
    #accuracy = accuracy_score(p.label_ids, pred)
    #roc_auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# Train the model
print("Training the model")
#trainer.train()

# Evaluate the model
print("Evaluating the model")
#evaluation_results = trainer.evaluate()

# Plot the results

# Store the metrics for each epoch
metrics_per_epoch = []
# Custom training loop to get metrics after each epoch
for epoch in range(training_args.num_train_epochs):
    trainer.train()
    print("Training the model")
    eval_metrics = trainer.evaluate()
    metrics_per_epoch.append(eval_metrics)
    print(f"Epoch {epoch+1} - {eval_metrics}")

# Final evaluation on the test set
print("Evaluating the model on test set")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
metrics_per_epoch.append(test_metrics)
print(f"Test - {test_metrics}")


# Plot the results
def plot_metrics(metrics_per_epoch):
    epochs = list(range(1, len(metrics_per_epoch)))
    epochs.append('Test')

    accuracy = [metrics['eval_accuracy'] for metrics in metrics_per_epoch]
    precision = [metrics['eval_precision'] for metrics in metrics_per_epoch]
    recall = [metrics['eval_recall'] for metrics in metrics_per_epoch]
    f1 = [metrics['eval_f1'] for metrics in metrics_per_epoch]
    auc = [metrics['eval_roc_auc'] for metrics in metrics_per_epoch]

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, accuracy, marker='o')
    plt.title('Accuracy')

    plt.subplot(3, 2, 2)
    plt.plot(epochs, precision, marker='o')
    plt.title('Precision')

    plt.subplot(3, 2, 3)
    plt.plot(epochs, recall, marker='o')
    plt.title('Recall')

    plt.subplot(3, 2, 4)
    plt.plot(epochs, f1, marker='o')
    plt.title('F1 Score')

    plt.subplot(3, 2, 5)
    plt.plot(epochs, auc, marker='o')
    plt.title('AUC')

    plt.tight_layout()
    plt.show()

plot_metrics(metrics_per_epoch)





