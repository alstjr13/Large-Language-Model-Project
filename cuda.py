import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print(torch.cuda.is_available())