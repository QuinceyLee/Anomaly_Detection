import gc
import sys

import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import time
from MyDataset import MyDataset, convert
from MyScaler import scale
from cnn_model import CNN
import pandas as pd
import numpy as np
from plot import get_confusion_matrix, plot_confusion_matrix
from utils.Logger import Logger

epochs = 10
batch_size = 10000
learning_rate = 0.01
n_raws = 5000000

cpu_num = 8  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def find_all_file(fold):
    for root, ds, fs in os.walk(fold):
        for f in fs:
            if not f.startswith('._'):
                yield f


sys.stdout = Logger('./out/result.log', sys.stdout)
sys.stderr = Logger('./out/result.log_file', sys.stderr)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将训练数据装入Loader中
filename = './test.csv'
train_dataset = MyDataset(filename, n_raws, shuffle=True)
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
print("开始训练主循环")
cnn.train()
for _ in range(epochs):
    tot_loss = 0.0
    tot_acc = 0.0
    train_pred = []
    train_trues = []
    train_dataset.initial()
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    for i, data in enumerate(train_iter):
        print(time.asctime(time.localtime(time.time())) + "正在进行第" + str(i) + '个batch')
        data_x, data_y = convert(data)
        # model.train()
        _, outputs = cnn(data_x)
        # _, preds = torch.max(outputs.data, 1)
        loss = loss_func(outputs, data_y)
        # print(loss)
        # 反向传播优化网络参数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 累加每个step的损失
        tot_loss += loss.data
        train_outputs = outputs.argmax(dim=1)
        train_pred.extend(train_outputs.detach().cpu().numpy())
        train_trues.extend(data_y.detach().cpu().numpy())
        # tot_acc += (outputs.argmax(dim=1) == train_label_batch).sum().item()
    sklearn_accuracy = accuracy_score(train_trues, train_pred)
    sklearn_precision = precision_score(train_trues, train_pred, average='micro')
    sklearn_recall = recall_score(train_trues, train_pred, average='micro')
    sklearn_f1 = f1_score(train_trues, train_pred, average='micro')
    print(
        "[sklearn_metrics] "
        "Epoch:{} "
        "loss:{:.4f} "
        "accuracy:{:.4f} "
        "precision:{:.4f} "
        "recall:{:.4f} "
        "f1:{:.4f}".format(
            epochs,
            tot_loss,
            sklearn_accuracy,
            sklearn_precision,
            sklearn_recall,
            sklearn_f1))
    torch.save(cnn.state_dict(), './out/cnn' + _ + '.pkl')
# dt_train = pd.read_csv('./new/train.csv', header=None)
torch.save(cnn.state_dict(), './out/cnn.pkl')

del train_dataset
gc.collect()
# test
dt_test = pd.read_csv('./test.csv', header=None)
dt_test = scale(dt_test.values)
x_test = dt_test[:, :-1].astype(float)
add_zeros = np.zeros(len(x_test))
x_test = np.c_[x_test, add_zeros]
x_test = np.reshape(x_test, (len(x_test), 4, 4))
y_test = dt_test[:, -1].astype(int)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)
x_test = torch.unsqueeze(x_test, dim=1)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

test_pred = []
test_trues = []
cnn.eval()

with torch.no_grad():
    for i, (test_data_batch, test_data_label) in enumerate(test_loader):
        _, test_outputs = cnn(test_data_batch)
        test_outputs = test_outputs.argmax(dim=1)
        test_pred.extend(test_outputs.detach().cpu().numpy())
        test_trues.extend(test_data_label.detach().cpu().numpy())
    sklearn_accuracy = accuracy_score(test_trues, test_pred)
    sklearn_precision = precision_score(test_trues, test_pred, average='micro')
    sklearn_recall = recall_score(test_trues, test_pred, average='micro')
    sklearn_f1 = f1_score(test_trues, test_pred, average='micro')
    print(classification_report(test_trues, test_pred))
    conf_matrix = get_confusion_matrix(test_trues, test_pred)
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix)
    print("[sklearn_metrics] "
          "accuracy:{:.4f} "
          "precision:{:.4f}"
          " recall:{:.4f} "
          "f1:{:.4f}".format
          (sklearn_accuracy,
           sklearn_precision,
           sklearn_recall,
           sklearn_f1))
