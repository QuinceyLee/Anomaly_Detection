import torch
from torch import nn
from torch.utils.data import DataLoader
from LSTMnet import LSTMnet

# Hyper Parameters
epochs = 5  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
batch_size = 1024
time_step = 10  # rnn 时间步数
input_size = 10  # rnn 每步输入值
hidden_size = 64
num_layers = 2
num_classes = 8
lr = 0.01  # learning rate

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)  # 在每个epoch开始的时候，对数据重新打乱进行训练。在这里其实没啥用，因为只训练了一次

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)

model = LSTMnet(input_size, hidden_size, num_layers, num_classes)  # 10*10，lstm的每个隐藏层64个节点，2层隐藏层

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# training and testing
for epoch in range(epochs):
    for iteration, (train_x, train_y) in enumerate(train_loader):  # train_x's shape (BATCH_SIZE,1,28,28)
        train_x = train_x.squeeze()  # after squeeze, train_x's shape (BATCH_SIZE,28,28),
        # 第一个28是序列长度，第二个28是序列中每个数据的长度。
        output = model(train_x)
        loss = criterion(output, train_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if iteration % 100 == 0:
            test_output = model(test_x)
            predict_y = torch.max(test_output, 1)[1].numpy()
            accuracy = float((predict_y == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('epoch:{:<2d} | iteration:{:<4d} | loss:{:<6.4f} | accuracy:{:<4.2f}'.format(epoch, iteration, loss,
                                                                                               accuracy))

# print 10 predictions from test data
test_out = model(test_x[:10])
pred_y = torch.max(test_out, dim=1)[1].data.numpy()
print('The predict number is:')
print(pred_y)
print('The real number is:')
print(test_y[:10].numpy())

# model = simple_lstm(input_size, hidden_size, num_layers, num_classes)
#
# # loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr)
# total_step = len(train_loader)
# for epoch in range(epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.reshape(-1, time_step, input_size)
#         labels = labels
#
#         # forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if i % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
