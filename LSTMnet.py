from torch import nn


# input_size 输入数据的大小 整个网络的输入input(seq_len, batch, input_size)
# hidden_size 隐藏层节点特征维度
# num_layers LSTM的层数
# bias ，网络是否设置偏置，默认是True.
# batch_first 如果这个参数为True 输入可以用input(batch, seq_len, input_size)
# dropout 随机丢失的比例，但是仅在多层LSTM的传递中使用。
# bidirectional，如果设置为True，则网络是双向LSTM，默认是False.


class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x's shape (batch_size, 序列长度, 序列中每个数据的长度)
        out, _ = self.lstm(x)  # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)  # 经过线性层后，out的shape为(batch_size, n_class)
        return out
