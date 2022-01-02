import pandas as pd

from dataProcess.encode.encode import input_names
from utils.find_file import find_all_file_csv
import os

sta_col_names = ['cata', 'num']

name = ['1-1.log.labeled',
        '3-1.log.labeled',
        '4-1.log.labeled',
        '5-1.log.labeled',
        '7-1-H.log.labeled',
        '7-1.log.labeled',
        '8-1.log.labeled',
        '9-1.log.labeled',
        '17-1.log.labeled',
        '20-1.log.labeled',
        '21-1.log.labeled',
        '33-1.log.labeled',
        '34-1.log.labeled',
        '35-1.log.labeled',
        '36-1.log.labeled',
        '39-1.log.labeled',
        '42-1.log.labeled',
        '43-1.log.labeled',
        '44-1.log.labeled',
        '48-1.log.labeled',
        '49-1.log.labeled',
        '52-1.log.labeled',
        '60-1.log.labeled'
        ]
sta_col = ['proto',
           'service',
           'conn_state',
           'history',
           "detailed-label"
           ]

root = '/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv_2'


def collect_sample(sample_dir, out_dir):
    for i in find_all_file_csv(sample_dir):
        print(i + '------------start')
        df = pd.read_csv(sample_dir + '/' + i, header=None, names=input_names)
        # 这里需要改input
        dirs = out_dir + '/' + os.path.splitext(i)[0]
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        for col in sta_col:
            dp = df[col].value_counts()
            dp.to_csv(dirs + '/' + col + '.csv')
        print(i + '---------------end')


# 读取数据
# r1 = pd.read_csv('/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv/1-1.log.labeled/service.csv', header=None, names=col_names)
# r2 = pd.read_csv('/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv/3-1.log.labeled/service.csv', header=None, names=col_names)
def merge_data(file_dir, out_dir):
    for i in sta_col:
        merge = pd.read_csv(file_dir + '/' + name[0] + '/' + i + '.csv', header=0, names=sta_col_names)
        for j in name[1:]:
            src = pd.read_csv(file_dir + '/' + j + '/' + i + '.csv', header=0, names=sta_col_names)
            merge = pd.concat([merge, src], axis=0, join='outer')
        merge = merge.groupby(sta_col_names[0], sort=False)[sta_col_names[1]].sum().reset_index()
        merge.to_csv(out_dir + '/' + i + '.csv')


# collect_sample(root, '/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/count_3')
# merge_data('/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/count_3',
#            '/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/count_merge_4')
# collect_sample('/Users/lee/PycharmProjects/pythonProject/new',
#                '/Users/lee/PycharmProjects/pythonProject/sta')
# merge_data('/Users/lee/PycharmProjects/pythonProject/sta',
#            '/Users/lee/PycharmProjects/pythonProject/merge')
