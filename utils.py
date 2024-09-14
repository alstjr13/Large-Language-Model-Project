import os
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


# Resulting data split:
# Train = 72 %
# Validation = 18 %
# Test = 10 %
def data_split(X, y):
    """
    Split data into 3 different dataset
    72% Train dataset
    18% Validation dataset
    10% Test dataset
    :param X: pd.DataFrame including review texts, string
    :param y: pd.DataFrame including labels, either 0 or 1
    :return: texts (train), labels (train), texts (validation), labels (validation), texts (test), labels (test)
    """

    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X_train, X_test, valid_train, valid_test = train_test_split(X_train_temp, X_test_temp, train_size=0.8, test_size=0.2)

    return X_train, X_test, valid_train, valid_test, y_train, y_test

def is_nan(x) -> bool:
    """
    Checks whether if the dataset includes a NaN value
    :param x: dataset to feed into the custom dataset
    :return: single bool, True -> x contains NaN, False -> x does not contain NaN
    """
    return x.isnull().values.any()