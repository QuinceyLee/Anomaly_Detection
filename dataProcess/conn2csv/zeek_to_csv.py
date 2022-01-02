from dataProcess.conn2csv import filed, process, log_csv
from utils.parsezeeklogs import ParseZeekLogs
from utils.find_file import find_all_file


for i in find_all_file(process):
    with open(log_csv + '/' + i + '.csv', mode='a') as outfile:
        for log_record in ParseZeekLogs(process + '/' + i, output_format="csv", safe_headers=False,
                                        fields=filed):
            if log_record is not None:
                outfile.write(log_record + "\n")
