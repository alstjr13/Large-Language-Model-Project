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

# Enable memory use
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings('ignore')

# ---------------------------------------PRE-PROCESSING-DATA---------------------------------------------------------

# Load dataset:
# reviewText                     (column 0): reviews in text... including incentivized and non-incentivized reviews
# incentivized_999               (column 1): - 0 : non-incentivized reviews
#                                            - 1 : incentivized reviews
filePath = "../data/updated_review_sample_for_RA.csv"
df = pd.read_csv(filePath)