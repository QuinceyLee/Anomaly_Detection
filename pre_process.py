import pandas as pd


# 划分数据
def divide_data():
    filepath = './new/merge2.csv'  # 数据文件路径
    data = pd.read_csv(filepath, header=None)
    train_data = data.sample(frac=0.8, random_state=8, axis=0)
    train_data.to_csv("./new/train/6.csv", header=None, index=None)
    test_data = data[~data.index.isin(train_data.index)]
    test_data.to_csv("./new/test/6.csv", header=None, index=None)


divide_data()
