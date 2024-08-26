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

#TODO

# Load sample data set with incentivized reviews as pd.DataFrame
filePath = '../data/updated_review_sample_for_RA.csv'
df = pd.read_csv(filePath)

# Delete any row that has NaN value (i.e. Clean)
df = df.dropna(subset=["reviewText"])  # Store dropped rows in an another .csv file for records

# Randomly select 100 incentivized reviews (Labelled with 1)
#                 100 not incentivized reviews (Labelled with 0)
notIncentivized = df[df["incentivized_999"] == 0].sample(n=10, random_state=42)
incentivized = df[df["incentivized_999"] == 1].sample(n=10, random_state=42)

# CHECK if there is NaN value in the extracted samples:
hasNaText = incentivized['reviewText'].isna().any()
hasNaLabel = incentivized['incentivized_999'].isna().any()
print(hasNaText)  # False
print(hasNaLabel)  # False
print("Shape of the non-incentivized:", notIncentivized.shape)  # (10, 3)
print("Shape of the incentivized:", incentivized.shape)  # (10, 3)

# Merge two dataframes and create size = (200 x 3) pd.DataFrame
newdf = pd.concat([notIncentivized, incentivized])
newdf = newdf.sample(frac=1, random_state=42).reset_index(drop=True)
print(newdf.shape)

# Drop newdf['incent_bert_highest_score_sent'] column
newdf = newdf.drop(['incent_bert_highest_score_sent'], axis=1)
print(newdf)
print(newdf.shape)

# Check for NaN values in the entire DataFrame
has_nan = newdf.isnull().values.any()
print(f"Does the DataFrame contain NaN values? {has_nan}")

# Split the data
X = newdf["reviewText"]
y = newdf["incentivized_999"]

# Default = 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Y Training Set Distribution: \n {pd.Series(y_train).value_counts()}")
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

# Metrics logs
logs = trainer.state.log_history

epochs = []
accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []

for log in logs:
    if "eval_accuracy" in log:
        epochs.append(log['epoch'])
        accuracy.append(log['eval_accuracy'])
        precision.append(log['eval_precision'])
        recall.append(log['eval_recall'])
        f1.append(log['eval_f1'])
        roc_auc.append(log['eval_roc_auc'])

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.plot(epochs, accuracy, label='Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')

plt.subplot(2,3,1)
plt.plot(epochs, precision, label='Precision', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision per Epoch')

plt.subplot(2,3,1)
plt.plot(epochs, recall, label='Recall', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall per Epoch')

plt.subplot(2,3,1)
plt.plot(epochs, accuracy, label='F1 Score', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')

plt.subplot(2,3,1)
plt.plot(epochs, accuracy, label='ROC_AUC', marker='o')
plt.xlabel('Epoch')
plt.ylabel('ROC_AUC')
plt.title('ROC_AUC per Epoch')

plt.tight_layout()
plt.show()