import os
import warnings

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import utils

#os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
notIncentivized = df[df['incentivized_999'] == 0].sample(n=10, random_state=42)
incentivized = df[df['incentivized_999'] == 1].sample(n=10, random_state=42)

# Combine random samples, and reset index and shuffle sample
df = pd.concat([notIncentivized, incentivized])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop unnecessary column
df = df.drop(['incent_bert_highest_score_sent'], axis=1)

X = df["reviewText"]
y = df["incentivized_999"]

# Split data to Train, Validation and Test (0.9 : 0.1 Ratio)
train_texts, train_labels, test_texts, test_labels = utils.data_split(X, y)

# Print number of labels in splited data
print(f"Training Set Distribution: \n {pd.Series(train_labels).value_counts()}")
print(f"Test Set Distribution: \n {pd.Series(test_labels).value_counts()}")

# Initialize BERT Large Model and BERT Large Tokenizer
model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
max_length = 4096

# Create ReviewDataset(Dataset), with encodings
trainDataset = utils.ReviewDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_length)
testDataset = utils.ReviewDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, max_length)

# GPU or CPU
model.to(device)

# --------------------------------------------FINE-TUNING---------------------------------------------------------------

training_args = TrainingArguments(
    output_dir='../results/longformer/longformer',
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
    logging_dir='../logs/longformer/longformer',
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
    model=model,
    args=training_args,
    train_dataset=trainDataset,
    eval_dataset=testDataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------------------------------------TRAINING----------------------------------------------------------

# Train pretrained model
print("Beginning to train the model")
trainer.train()

print("Training Done")

# Evaluate the trained model
print("Beginning to evaluate the model")
eval_metrics = trainer.evaluate()

# ---------------------------------------------------CROSS-VALIDATION---------------------------------------------------

crossval_results = []
crossval_accuracies = []
mean_epoch_accuracies = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross validation : split the data to 5 (90 / 5 = 18) --> Train : 72%, Validation : 18%
for fold, (train_index, valid_index) in enumerate(kf.split(train_texts)):
    print("DEBUG \n")
    print(f"Starting Fold {fold + 1} \n")
    fold_train_texts = train_texts.iloc[train_index].tolist()
    fold_train_labels = train_labels.iloc[train_index].tolist()
    fold_valid_texts = train_texts.iloc[valid_index].tolist()
    fold_valid_labels = train_labels.iloc[valid_index].tolist()

    model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    max_length = 4096

    # Create Datasets with
    crossval_train_dataset = utils.ReviewDataset(fold_train_texts, fold_train_labels, tokenizer, max_length)
    crossval_validation_dataset = utils.ReviewDataset(fold_valid_texts, fold_valid_labels, tokenizer, max_length)

    training_args_validation = TrainingArguments(
        output_dir='../results/longformer/longformerCrossValidation',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        # Alter:
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        adam_beta1=0.9,
        adam_beta2=0.99,

        # Fixed:
        logging_dir='../logs/longformer/longformerCrossValidation',
        # max_grad_norm= 15,
        num_train_epochs=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=5,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args_validation,
        train_dataset=crossval_train_dataset,
        eval_dataset=crossval_validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    model.to(device)

    # Begin training
    print("Cross validation train:")
    trainer.train()

    print(f"Fold {fold+1} Train Done")

    # Access model training history to extract accuracies per epoch
    logs_crossVal = trainer.state.log_history

    # Epoch 1, 2, 3, 4
    # [Accuracy1, Accuracy2, Accuracy3, Accuracy4]
    fold_epoch_accuracies = [log['eval_accuracy'] for log in logs_crossVal if 'eval_accuracy' in log]

    # Load trained model from Cross Validation
    # TODO: Need to alter checkpoint --> load the last checkpoint
    model_path_crossVal = "../results/longformer/longformerCrossValidation/checkpoint-60"
    model_trained_crossVal = AutoModelForSequenceClassification.from_pretrained(model_path_crossVal)

    test_trainer_crossVal = Trainer(model=model_trained_crossVal)

    predictions_output = test_trainer_crossVal.predict(crossval_validation_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=1)

    y_true = fold_valid_labels

    test_accuracy = accuracy_score(y_true, predictions)

    # [Accuracy1, Accuracy2, Accuracy3, Accuracy4, Test] for a Fold
    fold_epoch_accuracies.append(test_accuracy)

    # Combine the result from each fold to overall result
    # [[Fold1_Accuracy1, Fold1_Accuracy2, Fold1_Accuracy3, Fold1_Accuracy4, Fold1_Test], ... , [Fold5_Accuracy1, Fold5_Accuracy2, Fold5_Accuracy3, Fold5_Accuracy4, Fold5_Test]]
    crossval_results.append(fold_epoch_accuracies)

accuracy_results_crossVal = np.array(crossval_results)

# Resulting numpy list should be: [meanAccuracyCrossValEpoch1, meanAccuracyCrossValEpoch2, meanAccuracyCrossValEpoch3, meanAccuracyCrossValEpoch4, meanAccuracyCrossValTest]
mean_results_crossVal = np.mean(accuracy_results_crossVal, axis=0)

# ---------------------------------------------------METRICS------------------------------------------------------------
epochs = []
accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []
loss = []

# Get metrics from the evaluation
eval_accuracy = eval_metrics.get("eval_accuracy", None)
eval_precision = eval_metrics.get("eval_precision", None)
eval_recall = eval_metrics.get("eval_recall", None)
eval_f1 = eval_metrics.get("eval_f1", None)
eval_roc_auc = eval_metrics.get("eval_roc_auc", None)

# Metrics logs
logs = trainer.state.log_history

# Epoch 1, 2, 3, 4
for log in logs:
    if "eval_accuracy" in log:
        epoch_value = log['epoch']
        if epochs and epoch_value == epochs[-1]:
            continue
        epochs.append(epoch_value)
        accuracy.append(log['eval_accuracy'])
        precision.append(log['eval_precision'])
        recall.append(log['eval_recall'])
        f1.append(log['eval_f1'])
        roc_auc.append(log['eval_roc_auc'])
        loss.append(log['eval_loss'])

print("Epochs Metrics:")
print(epochs)
print(accuracy)
print(precision)
print(recall)
print(f1)
print(roc_auc)
print(loss)
print("\n")

print("Evaluation Metrics:")
print(f"Evaluation Accuracy: {eval_accuracy}")
print(f"Evaluation Precision: {eval_precision}")
print(f"Evaluation Recall: {eval_recall}")
print(f"Evaluation F1: {eval_f1}")
print(f"Evaluation ROC_AUC: {eval_roc_auc}")

model_path = '../results/bertWithoutCrossValidation/checkpoint-60'
model_trained = AutoModelForSequenceClassification.from_pretrained(model_path)

test_trainer = Trainer(model=model_trained)

predictions_output = test_trainer.predict(testDataset)
predictions = np.argmax(predictions_output.predictions, axis=1)

y_true = test_labels.tolist()

test_accuracy = accuracy_score(y_true, predictions)
test_precision = precision_score(y_true, predictions)
test_recall = recall_score(y_true, predictions)
test_f1 = f1_score(y_true, predictions)
test_roc_auc = roc_auc_score(y_true, predictions)

print("TEST ")
print(f"y_true: {y_true}")
print(f"predictions: {predictions}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")
print(f"Test ROC_AUC: {test_roc_auc}")

print("\n Append Test Results")
epochs.append("Test")
accuracy.append(test_accuracy)
precision.append(test_precision)
recall.append(test_recall)
f1.append(test_f1)
roc_auc.append(test_roc_auc)

cm = confusion_matrix(y_true, predictions)

# Predictions
predictions = trainer.predict(testDataset)

# Load trained-model

model_path = "../results/bigbird/bigbird/checkpoint-60"
model_trained = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define test trainer
test_trainer = Trainer(model_trained)

raw_pred, _,_ = test_trainer.predict(testDataset)

y_pred = np.argmax(raw_pred, axis=1)
print("Prediction DEBUG")
print(y_pred)

# Loss function
predictions, labels, _ = test_trainer.predict(testDataset)
predictions = torch.tensor(predictions)
labels = torch.tensor(labels)

# Plotting
plt.figure(figsize=(12,8))

# Accuracy Plot
plt.subplot(2,3,1)
plt.plot(epochs, accuracy, marker='o')
plt.plot(epochs, mean_results_crossVal, marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')

# Precision Plot
plt.subplot(2,3,2)
plt.plot(epochs, precision, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision per Epoch')

# Recall Plot
plt.subplot(2,3,3)
plt.plot(epochs, recall, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall per Epoch')

# F1 Score Plot
plt.subplot(2,3,4)
plt.plot(epochs, f1, marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')

# ROC_AUC Plot
plt.subplot(2,3,5)
plt.plot(epochs, roc_auc, marker='o')
plt.xlabel('Epoch')
plt.ylabel('ROC_AUC')
plt.title('ROC_AUC per Epoch')

# Heat map
plt.subplot(2,3,6)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()

plt.savefig('plot_of_metrics_longformer.png')