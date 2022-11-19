import logging
import os
import time

import psutil

logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log',encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)  # 设置日志的级别
# fh.setFormatter(formatter)#设置的日志的输出
ch.setFormatter(formatter)
# logger.addHandler(fh) #logger对象可以添加多个fh和ch对象
logger.addHandler(ch)


def getNetworkData():
    # 获取网卡流量信息
    recv = {}
    sent = {}
    data = psutil.net_io_counters(pernic=True)
    interfaces = data.keys()
    for interface in interfaces:
        recv.setdefault(interface, data.get(interface).bytes_recv)
        sent.setdefault(interface, data.get(interface).bytes_sent)
    return interfaces, recv, sent


if __name__ == '__main__':
    base_sent = psutil.net_io_counters().bytes_sent
    base_recv = psutil.net_io_counters().bytes_recv
    while True:
        logger.info('network(MB) - sent:{},recv:{}'.format((psutil.net_io_counters().bytes_sent - base_sent)/1048576,
                                                    (psutil.net_io_counters().bytes_recv - base_recv)/1048576))
        time.sleep(2)
    # test_m = multiprocessing.Process(target=get_pid_memory, args=(test_p.pid,))
    # test_m.start()
