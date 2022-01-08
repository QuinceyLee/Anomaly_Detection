import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from MyScaler import scale
from cnn_model import CNN
import pandas as pd
import numpy as np
from plot import get_confusion_matrix, plot_confusion_matrix

cnn = CNN()
cnn.load_state_dict(torch.load('./out/cnn.pkl'))
test_pred = []
test_trues = []
cnn.eval()
test_file_name = [
    '3.csv',
    '4.csv',
    '12.csv',
    '25.csv',
    '26.csv',
    '33.csv',
    '34.csv',
    '37.csv'
]
test_folder = './test/'
batch_size = 10000
with torch.no_grad():
    for file_name in test_file_name:
        dt_test = pd.read_csv(test_folder + file_name, header=None)
        dt_test = scale(dt_test.values)
        x_test = dt_test[:, :-1].astype(float)
        add_zeros = np.zeros(len(x_test))
        x_test = np.c_[x_test, add_zeros]
        x_test = np.reshape(x_test, (len(x_test), 4, 4))
        y_test = dt_test[:, -1].astype(int)
        x_test, y_test = torch.FloatTensor(x_test), torch.IntTensor(y_test)
        x_test = torch.unsqueeze(x_test, dim=1)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size)
        for i, (train_data_batch, test_data_label) in enumerate(test_loader):
            _, test_outputs = cnn(train_data_batch)
            test_outputs = test_outputs.argmax(dim=1)
            test_pred.extend(test_outputs.detach().cpu().numpy())
            test_trues.extend(test_data_label.detach().cpu().numpy())
            print('文件' + file_name + '        第' + str(i) + '个batch')
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
