import os
import warnings
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, \
    precision_score, recall_score, f1_score
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings('ignore')

# Load sample dataset:
# reviewText                     (column 0): reviews in text... including incentivized and non-incentivized reviews
# incentivized_999               (column 1): - 0 : non-incentivized reviews
#                                            - 1 : incentivized reviews
filePath = "../../data/updated_review_sample_for_RA.csv"
df = pd.read_csv(filePath)

# Delete any row that has NaN value
df = df.dropna(subset=["reviewText"])

# Take random samples from the dataset (.csv file)
notIncentivized = df[df['incentivized_999'] == 0].sample(n=300, random_state=42)
incentivized = df[df['incentivized_999'] == 1].sample(n=300, random_state=42)

# Check if there exist NaN value in the extracted samples:
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
# texts:  reviews of the users (either incentivized or unincentivized - labelled with 0 or 1)
# labels: corresponding label value --> 0 or 1
# tokenizer: BERT-Large Tokenizer
# max_length: set default to 512
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

#optimizer: Default --> AdamW
training_args = TrainingArguments(
    output_dir='./results/bert',
    overwrite_output_dir= True,
    do_train= True,
    do_eval= True,

    # Alter:
    adam_beta1=0.9,
    adam_beta2=0.999,
    learning_rate=3e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    # Fixed:
    logging_dir='./logs/bert',
    # num_train_epochs = 4 ~ 5
    # BERT 논문 권장: 2 ~ 4
    num_train_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=5,
    load_best_model_at_end=True,
)

"""
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = (2 * Precision * Recall) / (Precision + Recall)
"""
def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)

    print(f"Labels: {labels}")
    print(f"Predictions: {preds}")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average="weighted")
    roc_auc = roc_auc_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
print("Beginning to train the model")
trainer.train()

# Evaluate (Test)
print("Beginning to evaluate the model")
eval_metrics = trainer.evaluate()

# Metrics from the testing stage
eval_accuracy = eval_metrics.get("eval_accuracy", None)
eval_precision = eval_metrics.get("eval_precision", None)
eval_recall = eval_metrics.get("eval_recall", None)
eval_f1 = eval_metrics.get("eval_f1", None)
eval_roc_auc = eval_metrics.get("eval_roc_auc", None)

# Metrics logs
logs = trainer.state.log_history

epochs = []
accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []
loss = []

for log in logs:
    if "eval_accuracy" in log:
        epochs.append(log['epoch'])
        accuracy.append(log['eval_accuracy'])
        precision.append(log['eval_precision'])
        recall.append(log['eval_recall'])
        f1.append(log['eval_f1'])
        roc_auc.append(log['eval_roc_auc'])
        #loss.append(log['eval_loss'])

epochs.append("Test")
accuracy.append(eval_accuracy)
precision.append(eval_precision)
recall.append(eval_recall)
f1.append(eval_f1)
roc_auc.append(eval_roc_auc)

print(epochs)
print(accuracy)
print(precision)
print(recall)
print(f1)
print(roc_auc)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.plot(epochs, accuracy, label='Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')

plt.subplot(2,3,2)
plt.plot(epochs, precision, label='Precision', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision per Epoch')

plt.subplot(2,3,3)
plt.plot(epochs, recall, label='Recall', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall per Epoch')

plt.subplot(2,3,4)
plt.plot(epochs, f1, label='F1 Score', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')

plt.subplot(2,3,5)
plt.plot(epochs, roc_auc, label='ROC_AUC', marker='o')
plt.xlabel('Epoch')
plt.ylabel('ROC_AUC')
plt.title('ROC_AUC per Epoch')

plt.tight_layout()
plt.show()