import gc
import sys
from random import shuffle

import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import time
from MyScaler import scale
from cnn_model import CNN
import pandas as pd
import numpy as np
from plot import get_confusion_matrix, plot_confusion_matrix
from utils.Logger import Logger

# batch_size = 10000
from utils.find_file import find_all_file_csv

for i in find_all_file_csv('../divide'):
    dt_test = pd.read_csv('../divide/' + i, header=None, usecols=[15])
    print(i)
    print(dt_test.value_counts())

