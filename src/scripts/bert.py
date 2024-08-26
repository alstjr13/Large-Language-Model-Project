import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
#from sklearn.model_selection import cross_val_score
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Load sample data set with incentivized reviews as pd.DataFrame
filePath = '../../data/updated_review_sample_for_RA.csv'
df = pd.read_csv(filePath)

# -------- 1. Preprocess the data ----------#
# Delete any row that has NaN value (i.e. Clean)
df = df.dropna(subset=["reviewText"])  # Store dropped rows in an another .csv file for records

# Randomly select 100 incentivized reviews (Labelled with 1)
#                 100 not incentivized reviews (Labelled with 0)
notIncentivized = df[df["incentivized_999"] == 0].sample(n=100, random_state=42)
incentivized = df[df["incentivized_999"] == 1].sample(n=100, random_state=42)

# Merge two dataframes and create size = (200 x 3) pd.DataFrame
newdf = pd.concat([notIncentivized, incentivized])
newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop newdf['incent_bert_highest_score_sent'] column
newdf = newdf.drop(['incent_bert_highest_score_sent'], axis=1)

# Check for NaN values in the entire DataFrame
has_nan = newdf.isnull().values.any()
print(f"Does the DataFrame contain NaN values? {has_nan}")

# Split the data
X = list(newdf["reviewText"])
y = list(newdf["incentivized_999"])

# Default = 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Y Training Set Distribution: \n {pd.Series(y_train).value_counts()}")
print(f"Y Test Set Distribution: \n {pd.Series(y_test).value_counts()}")

# Initialize BERT-large model and tokenizer with maxlength = 512
model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
max_length = 512

X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True, return_attention_mask=True,max_length=max_length, return_tensors="pt")
X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True, return_attention_mask=True,max_length=max_length, return_tensors='pt')

print("Sample data and labels:")
for i in range(5):
    print(f"Review: {X_train[i]}")
    print(f"Label: {y_train[i]}")

# --------------------------------------------------------------------------------------------------------------------------------

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

# --------------------------------------------------------------------------------------------------------------------------------

train_dataset = ReviewsDataset(X_train_tokenized, y_train)
test_dataset = ReviewsDataset(X_test_tokenized, y_test)

# Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#optimizer: Default --> AdamW
training_args = TrainingArguments(
    output_dir='./results/bert',
    overwrite_output_dir= True,
    do_train= True,
    do_eval= True,

    # Alter:
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,

    # Fixed:
    logging_dir='./logs/bert',
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500,
    logging_steps=5,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
)

#TODO: Fine-tune pretrained model
# --------- 2. Fine-Tune Pretrained Model -------- #
def compute_metrics(p):
    pred, labels = p
    probs = torch.nn.functional.softmax(torch.tensor(pred), dim=1).numpy()
    pred_labels = torch.argmax(torch.tensor(pred), axis=1)

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    roc_auc = roc_auc_score(labels, probs[:,1])

    print(f"Accuracy: {accuracy}, \n"
          f"Precision: {precision}, \n"
          f"Recall: {recall}, \n"
          f"F1 Score: {f1}, \n"
          f"AUC: {roc_auc} \n")
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
    compute_metrics=compute_metrics,
)


# Train
print("Beginning to train the model")
trainer.train()

# Evaluate
print("Beginning to evaluate the model")
eval_metrics = trainer.evaluate()

print("Evaluate, metrics")
print(eval_metrics)

