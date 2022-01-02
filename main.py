# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import csv


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def tex2string(str):
    with open(str + '.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        # 读要转换的txt文件，文件每行各词间以字符分隔
        with open(str + '.txt', 'r', encoding='utf-8') as filein:
            for line in filein:
                line_list = line.strip('\n').split(';')  # 我这里的数据之间是以 ; 间隔的
                spamwriter.writerow(line_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tex2string("household_power_consumption")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
