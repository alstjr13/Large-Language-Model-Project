import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, \
    TrainerState, TrainerControl
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_score, \
    recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

#TODO

# Load sample data set with incentivized reviews as pd.DataFrame
filePath = '../data/updated_review_sample_for_RA.csv'
df = pd.read_csv(filePath)

# Delete any row that has NaN value (i.e. Clean)
df = df.dropna(subset=["reviewText"])               # Store dropped rows in an another .csv file for records

# Randomly select 100 incentivized reviews (Labelled with 1)
#                 100 not incentivized reviews (Labelled with 0)
notIncentivized = df[df["incentivized_999"] == 0].sample(n=30, random_state=42)
incentivized = df[df["incentivized_999"] == 1].sample(n=30, random_state=42)

# CHECK if there is NaN value in the extracted samples:
hasNaText = incentivized['reviewText'].isna().any()
hasNaLabel = incentivized['incentivized_999'].isna().any()
print(hasNaText)                                                    # False
print(hasNaLabel)                                                   # False
print("Shape of the non-incentivized:", notIncentivized.shape)      # (10, 3)
print("Shape of the incentivized:", incentivized.shape)             # (10, 3)

# Merge two dataframes and create size = (200 x 3) pd.DataFrame
newdf = pd.concat([notIncentivized, incentivized])
newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)
print(newdf.shape)

#newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)
#print(newdf.shape)                                                  # (20, 3)

# Drop newdf['incent_bert_highest_score_sent'] column
newdf = newdf.drop(['incent_bert_highest_score_sent'], axis=1)
print(newdf)
print(newdf.shape)

#print(newdf)
#print(newdf.shape)                                                  # (20, 2)

# Check for NaN values in the entire DataFrame
has_nan = newdf.isnull().values.any()
print(f"Does the DataFrame contain NaN values? {has_nan}")

# Split the data
X = newdf["reviewText"]
y = newdf["incentivized_999"]

# Default = 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

#print(f"X Training Set Distribution: \n {pd.Series(X_train).value_counts()}")
print(f"Y Training Set Distribution: \n {pd.Series(y_train).value_counts()}")
#print(f"X Test Set Distribution: \n {pd.Series(X_test).value_counts()}")
print(f"Y Test Set Distribution: \n {pd.Series(y_test).value_counts()}")

# --------------------------------------------------------------------------------------------------------------------------------

# Create a ReviewDataset with inputs:
# 1. texts:  reviews of the users (either incentivized or unincentivized - labelled with 0 or 1)
# 2. labels: corresponding label value --> 0 or 1
# 3. tokenizer: BERT-Large Tokenizer
# 4. max_length: set default to 512
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = []
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.metrics.append(metrics)

# --------------------------------------------------------------------------------------------------------------------------------

# Initialize BERT-large model and tokenizer with maxlength = 512
model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
max_length = 512

# Datasets to feed into BERT-Large: ReviewDataset(Dataset)
train_dataset = ReviewsDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length)
test_dataset = ReviewsDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length)

# Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storing metrics
metrics_callback = MetricsCallback()

#optimizer: Default --> AdamW
training_args = TrainingArguments(
    output_dir='./results/bert',
    overwrite_output_dir= True,
    do_train= True,
    do_eval= True,

    # Alter:
    adam_beta1=0.9,
    adam_beta2=0.999,
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,

    # Fixed:
    logging_dir='./logs/bert',
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=5,
    load_best_model_at_end=True,
)

#TODO
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    #probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)
    pred_labels = torch.argmax(torch.tensor(predictions), axis=1).numpy()

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    roc_auc = roc_auc_score(labels, pred_labels)

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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

metrics = []

metrics_dict = {
    "loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}

# Train
print("Beginning to train the model")
trainer.train()

# Evaluate
print("Beginning to evaluate the model")
eval_metrics = trainer.evaluate()

metrics.append(eval_metrics)
print(metrics)

#for epoch, epoch_metrics in enumerate(metrics_callback):
#    print(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

"""
#TODO
metrics_kfold = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")

    # Create training and validation dataset
    X_train_fold, X_val_fold = X_train.iloc[train_idx].tolist(), X_train.iloc[val_idx].tolist()
    y_train_fold, y_val_fold = y_train.iloc[train_idx].tolist(), y_train.iloc[val_idx].tolist()

    train_dataset_fold = ReviewsDataset(X_train_fold, y_train_fold, tokenizer, max_length)
    val_dataset_fold = ReviewsDataset(X_val_fold, y_val_fold, tokenizer, max_length)

    model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)

    training_args_fold = TrainingArguments(
        output_dir=f"../results/bert_fold_{fold}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        # Alter:
        adam_beta1=0.9,
        adam_beta2=0.999,
        learning_rate=1e-7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        # Fixed:
        logging_dir=f'../logs/bert_fold_{fold}',
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=5,
        load_best_model_at_end=True,
    )

    trainer_fold = Trainer(
        model=model,
        args=training_args_fold,
        train_dataset=train_dataset_fold,
        eval_dataset=val_dataset_fold,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer_fold.train()

    eval_metrics_fold = trainer_fold.evaluate()
    print(f"Metrics for fold {fold + 1}: {eval_metrics_fold}")
    metrics_kfold.append(eval_metrics_fold)
"""


"""
def plot_metrics(metrics_per_epoch):
    epochs = list(range(1, len(metrics_per_epoch)))
    epochs.append('Test')

    accuracy = [metrics['eval_accuracy'] for metrics in metrics_per_epoch]
    precision = [metrics['eval_precision'] for metrics in metrics_per_epoch]
    recall = [metrics['eval_recall'] for metrics in metrics_per_epoch]
    f1 = [metrics['eval_f1'] for metrics in metrics_per_epoch]
    auc = [metrics['eval_roc_auc'] for metrics in metrics_per_epoch]

    x_axis = ["epoch1", "epoch2", "epoch3", "test"]

    plt.figure(figsize=(20, 10))

    plt.subplot(4, 2, 1)
    plt.plot(epochs, accuracy)

    plt.title('Accuracy')

    plt.subplot(4, 2, 2)
    plt.plot(epochs, precision)
    plt.title('Precision')

    plt.subplot(4, 2, 3)
    plt.plot(epochs, recall)
    plt.title('Recall')

    plt.subplot(4, 2, 4)
    plt.plot(epochs, f1)
    plt.title('F1 Score')

    plt.subplot(4, 2, 5)
    plt.plot(epochs, auc)
    plt.title('AUC')

    plt.tight_layout()
    plt.show()

plot_metrics(metrics_per_epoch)
"""
