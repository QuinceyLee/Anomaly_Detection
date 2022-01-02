from dataProcess.conn2csv import filed
from utils.parsezeeklogs import ParseZeekLogs

with open('out.csv', mode='a') as outfile:
    for log_record in ParseZeekLogs('/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/big/33-1-n.log.labeled',
                                    output_format="csv", safe_headers=False, fields=filed):
        if log_record is not None:
            outfile.write(log_record + "\n")
