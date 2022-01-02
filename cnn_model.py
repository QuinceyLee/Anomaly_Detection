import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,4，4]
            nn.Conv2d(
                in_channels=1,  # 输入图片的高度
                out_channels=80,
                kernel_size=(2, 2),
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫一次
                padding=1,  # 给图外边补上0
            ),
            # 经过卷积层 输出[80,5,5] 传入池化层
            nn.LeakyReLU()
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=80,
                out_channels=40,
                kernel_size=(2, 2),
                stride=(1, 1),
                padding=0
            ),
            # 经过卷积 输出[40, 4, 4]
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=40,  # 同上
                out_channels=20,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            # 经过卷积 输出[20, 4, 4] 传入池化层
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,  # 同上
                out_channels=30,
                kernel_size=(2, 2),
                stride=(1, 1),
                padding=0
            ),
            # 经过卷积 输出[30, 3, 3] 传入池化层
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Linear(30 * 3 * 3, 100)
        self.softmax = nn.Softmax(dim=1)

        # 输出层
        self.output = nn.Linear(in_features=100, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 保留batch
        x = self.fc1(x)
        x = self.softmax(x)
        output = self.output(x)  # 输出[8]
        return x, output
