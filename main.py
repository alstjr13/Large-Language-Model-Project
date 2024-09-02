import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf


# Checking for the GPU
device_name = tf.test.gpu_device_name()
print(device_name)

# Project with BERT large, Bigbird-RoBERTa-large, Longformer

# TODO:
# Make 5 figures of epoch = 1,2,3 and test of the following criterias:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1 Score
# 5. AUC
# Epoch 1,2,3 test

#if __name__ == "__main__":
#    print("hello world")