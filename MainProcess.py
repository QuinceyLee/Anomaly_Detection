# # 'tcpdump -i eth0 -s0 -C 5 -Z root -w %Y_%m%d_%H%M_%S.pcap'
#
# # 'zeek -Cr 2019-01-10-14-34-38-192.168.1.197.pcap'
#
# # 取出conn.log 变成csv
#
# # csv发送mq 存储
# # 预处理---->调模型---->得出结果----->存储
# import subprocess, fcntl, os
# def tcpdump():
#
#     cmd1 = ['tcpdump', '-i', 'eth0', '-n', '-B', '4096', '-s', '0', '-w', '%Y_%m%d_%H%M_%S.pcap']
#     p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
#
#     flags = fcntl.fcntl(p2.stdout.fileno(), fcntl.F_GETFL)
#     fcntl.fcntl(p2.stdout.fileno(), fcntl.F_SETFL, (flags | os.O_NDELAY | os.O_NONBLOCK))
#     return p2
#
#
# def poll_tcpdump(proc):
#     # print 'poll_tcpdump....'
#     import select
#     txt = None
#     while True:
#     # wait 1/10 second
#     readReady, _, _ = select.select([proc.stdout.fileno()], [], [], 0.1)
#     if not len(readReady):
#         break
#     try:
#         for line in iter(proc.stdout.readline, ""):
#             if txt is None:
#                 txt = ''
#         txt += line
#     except IOError:
#         print
#         'data empty...'
#         pass
#     break
#     return txt
#
#
# proc = tcpdump()
# while True:
#     text = poll_tcpdump(proc)
#     if text:
#         print
#         '>>>> ' + text
