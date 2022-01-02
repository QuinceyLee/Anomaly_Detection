import torch
import torch.utils.data as Data
import random
import numpy as np
from MyScaler import scale
import pandas as pd


def convert(data_in):
    result = None
    for d in data_in:
        if d == '':
            break
        if result is None:
            result = np.array(d.split(','))
        else:
            temp = np.array(d.split(','))
            result = np.c_[result, temp]
    result = result.astype(float)
    result = scale(result.T)
    x = result[:, :-1].astype(float)
    add_zeros = np.zeros(len(x))
    x = np.c_[x, add_zeros]
    x = np.reshape(x, (len(x), 4, 4))
    y = result[:, -1].astype(int)
    x, y = torch.FloatTensor(x), torch.LongTensor(y)
    x = torch.unsqueeze(x, dim=1)
    return x, y


class MyDataset(Data.Dataset):
    def __init__(self, file_path, n_raws, shuffle=False):
        """
        file_path: the path to the dataset file
        n_raws: each time put n_raws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        file_raws = 0
        # get the count of all samples
        with open(file_path, 'r') as f:
            for _ in f:
                file_raws += 1
        self.file_path = file_path
        self.file_raws = file_raws
        self.n_raws = n_raws
        self.shuffle = shuffle

    def initial(self):
        self.f_input = open(self.file_path, 'r')
        self.samples = list()
        # put nraw samples into memory
        for _ in range(self.n_raws):
            data = self.f_input.readline()  # data contains the feature and label
            if data:
                self.samples.append(data)
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return self.file_raws

    def __getitem__(self, item):
        idx = self.index[0]
        data = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:
            # all the samples in the memory have been used, need to get the new samples
            self.samples.clear()
            for _ in range(self.n_raws):
                data = self.f_input.readline()  # data contains the feature and label
                if data:
                    self.samples.append(data)
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)
        return data
