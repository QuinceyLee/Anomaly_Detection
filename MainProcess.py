'tcpdump -i eth0 -s0 -C 5 -Z root -w %Y_%m%d_%H%M_%S.pcap'

'zeek -Cr 2019-01-10-14-34-38-192.168.1.197.pcap'

# 取出conn.log 变成csv

# csv发送mq 存储
# 预处理---->调模型---->得出结果----->存储
