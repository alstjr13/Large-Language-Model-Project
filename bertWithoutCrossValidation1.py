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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

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

# Split data to Train, Validation and Test (0.9 : 0.1 Ratio)
train_texts, test_texts, train_labels, test_labels = utils.data_split(X, y)

# Print number of labels in splited data
print(f"Training Set Distribution: \n {pd.Series(train_labels).value_counts()}")
print(f"Test Set Distribution: \n {pd.Series(test_labels).value_counts()}")

# Initialize BERT Large Model and BERT Large Tokenizer
model = BertForSequenceClassification.from_pretrained("bert-large-cased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
max_length = 512

# GPU or CPU
#model.to(device)

# Create ReviewDataset(Dataset), with encodings
trainDataset = utils.ReviewDataset(train_texts.tolist(), train_labels.tolist(), tokenizer=tokenizer, max_length=max_length)
testDataset = utils.ReviewDataset(test_texts.tolist(), test_labels.tolist(), tokenizer=tokenizer, max_length=max_length)

# --------------------------------------------FINE-TUNING---------------------------------------------------------------

training_args = TrainingArguments(
    output_dir='../results/bert/bertWithoutCrossValidation',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,

    # Alter
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    adam_beta1=0.9,
    adam_beta2=0.99,

    # Fixed
    logging_dir='../logs/bert/bertWithoutCrossValidation',
    num_train_epochs=4,
    eval_strategy='epoch',
    save_strategy='epoch',
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=5,
    load_best_model_at_end=True,
)



def compute_metrics(p):
    """
    Computes the accuracy, precision, recall, F1, ROC_AUC of the input predictions
    :param p: predictions
    :return: accuracy, precision, recall, f1, roc_auc
    """
    labels = p.label_ids
    preds = p.predictions.argmax(-1)

    # For DEBUGGING
    print(f"Labels: {labels} \n")
    print(f"Predictions: {preds}")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # For DEBUGGING
    print(f"Accuracy: {accuracy}, \n"
          f"Precision: {precision}, \n"
          f"Recall: {recall}, \n"
          f"F1 Score: {f1}, \n"
          f"AUC: {roc_auc} \n"
          f"True Positives: {tp} \n"
          f"False Positives: {fp} \n"
          f"True Negatives: {tn} \n"
          f"False Negatives: {fn}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }




# Initialize Trainer to train the pre-trained model
trainer = Trainer(
    model=model,                        # BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=testDataset,
    tokenizer=tokenizer,                # BertTokenizer.from_pretrained('bert-large-cased')
    compute_metrics=compute_metrics,
)


# ----------------------------------------------------TRAINING----------------------------------------------------------
# Train pretrained model
#print("Beginning to train the model")
#trainer.train()

#print("Training Done")

# Evaluate the trained model
#print("Beginning to evaluate the model")
#eval_metrics = trainer.evaluate()

# ---------------------------------------------------CROSS-VALIDATION---------------------------------------------------
from sklearn.model_selection import KFold, cross_val_score

crossval_results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross validation : split the data to 5 (90 / 5 = 18) --> Train : 72%, Validation : 18%
for fold, (train_index, valid_index) in enumerate(kf.split(train_texts)):
    print("DEBUG \n")
    print(f"Starting Fold {fold + 1} \n")
    fold_train_texts = train_texts.iloc[train_index].tolist()
    fold_train_labels = train_labels.iloc[train_index].tolist()
    fold_valid_texts = train_texts.iloc[valid_index].tolist()
    fold_valid_labels = train_labels.iloc[valid_index].tolist()

    crossval_train_dataset = utils.ReviewDataset(fold_train_texts, fold_train_labels, tokenizer=tokenizer, max_length=max_length)
    crossval_validation_dataset = utils.ReviewDataset(fold_valid_texts, fold_valid_labels, tokenizer=tokenizer, max_length=max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
    )

"""
for fold, (train_index, valid_index) in enumerate(kf.split(train_texts)):
    print(f"Begin Fold {fold + 1}")
    fold_train_texts = train_texts.iloc[train_index].tolist()
    fold_train_labels = train_labels.iloc[train_index].tolist()
    fold_valid_texts = train_texts.iloc[valid_index].tolist()
    fold_valid_labels = train_labels.iloc[valid_index].tolist()

    fold_train_dataset = utils.ReviewDataset(fold_train_texts, fold_train_labels, tokenizer=tokenizer, max_length=max_length)
    fold_valid_dataset = utils.ReviewDataset(fold_valid_texts, fold_valid_labels, tokenizer=tokenizer, max_length=max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fold_train_dataset,
        eval_dataset=fold_valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate the model for the current fold
    trainer.train()
    fold_eval_metrics = trainer.evaluate()

    # Store the results of the fold
    crossval_results.append(fold_eval_metrics)
"""




# ---------------------------------------------------METRICS------------------------------------------------------------
epochs = []

# Get metrics from the evaluation

"""
eval_accuracy = eval_metrics.get("eval_accuracy", None)
eval_precision = eval_metrics.get("eval_precision", None)
eval_recall = eval_metrics.get("eval_recall", None)
eval_f1 = eval_metrics.get("eval_f1", None)
eval_roc_auc = eval_metrics.get("eval_roc_auc", None)

# Metrics logs
logs = trainer.state.log_history
"""








