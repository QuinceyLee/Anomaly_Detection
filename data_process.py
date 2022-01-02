import pandas as pd
import numpy as np
import sys
import sklearn

print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)
col_names = ["id.orig_p",
             "id.resp_p",
             "proto",
             "service",
             "duration",
             "orig_bytes",
             "resp_bytes",
             "conn_state",
             "local_orig",
             "local_resp",
             "missed_bytes",
             "history",
             "orig_pkts",
             "orig_ip_bytes",
             "resp_pkts",
             "resp_ip_bytes",
             "tunnel_parents",
             "label",
             "detailed-label"
             ]
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
df = pd.read_csv("/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv/data/33-1.log.labeled.csv", header=None, names=col_names)
print('Dimensions of the Training set:', df.shape)
print(df.head(5))
print(df.describe())
print('Label distribution Training set:')

# sta_col = ["proto",
#            "service",
#            "conn_state",
#            "local_orig",
#            "local_resp",
#            "missed_bytes",
#            "history",
#            "tunnel_parents",
#            ]
# for col in sta_col:
#     dp = df[col].value_counts()
#     print(type(dp))
