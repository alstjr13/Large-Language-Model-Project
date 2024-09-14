from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


# Resulting data split:
# Train = 72 %
# Validation = 18 %
# Test = 10 %
def data_split(X, y):

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X_train, X_test, valid_train, valid_test = train_test_split(X_train_temp, X_test_temp, train_size=0.8, test_size=0.2)

    # returns:
    # texts (train), labels (train), texts (validation), labels (validation), texts (test), labels (test)
    return X_train, X_test, valid_train, valid_test, y_train, y_test