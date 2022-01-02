import sys

import torch

from MyDataset import MyDataset
import torch.utils.data as Data
import numpy as np
import pandas as pd

from utils.Logger import Logger

data_path = "./new/44-1.log.labeled.csv"
batch_size = 5
n_raws = 12
epoch = 3
train_dataset = MyDataset(data_path, n_raws, shuffle=False)

# def convert(data_in):
#     result = None
#     for d in data_in:
#         if result is None:
#             result = np.array(d.split(','))
#         else:
#             temp = np.array(d.split(','))
#             result = np.c_[result, temp]
#     return result.T


# for _ in range(epoch):
#     train_dataset.initial()
#     train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size)
#     for _, data in enumerate(train_iter):
#         data = convert(data)
#         data = torch.Tensor(data.astype(float))
#         print(data)
#         print(type(data))

# data = [
#     '1545459136.586083,36097,37215,0,0,0.0,0,0,1,0,1,1,40,0,0,4',
#     '1551396595.645932,36616,23,0,0,3.097479,0,0,1,0,1,6,360,0,0,2',
#     '1545453976.534286,36097,37215,0,0,0.0,0,0,1,0,1,1,40,0,0,4',
#     '1532558590.005984,31825,23,0,0,0.0,0,0,1,0,1,1,40,0,0,2',
#     '1551397034.77089,30535,8081,0,0,0.000256,0,0,1,0,1,4,160,0,0,2',
# ]
# res = convert(data)
# print(type(res))
sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('a.log_file', sys.stderr)
df = pd.read_csv("./new/44-1.log.labeled.csv", header=None, usecols=[10, 11, 12, 13, 14])
print('mean = ', np.mean(df))
print('std = ', np.std(df))
