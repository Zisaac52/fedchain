import datetime
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


def get_pid_memory(pid):
    """
    根据进程号来获取进程的内存大小
    :param pid: 进程id
    :return: pid内存大小/MB
    """
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024


def get_process_memory(process_name):
    """
    获取同一个进程名所占的所有内存
    :param process_name:进程名字
    :return:同一个进程名所占的所有内存/MB
    """
    total_mem = 0
    for i in psutil.process_iter():
        if i.name() == process_name:
            total_mem += get_pid_memory(i.pid)
    print('{:.2f} MB'.format(total_mem))
    return total_mem


def test(n):
    pid = os.getpid()
    print("Test PID:{}".format(pid))

    for i in range(n):
        print("Test i{}...".format(i))
    return pid


def getProcessInfo(p):
    """取出指定进程占用的进程名，进程ID，进程实际内存, 虚拟内存,CPU使用率
    """
    try:
        cpu = p.cpu_percent(interval=0)
        rss, vms, mms = 0, 0, 0
        print(p.memory_info())
        name = p.name
        pid = p.pid
    except psutil.NoSuchProcess as e:
        name = "Closed_Process"
        pid = 0
        rss = 0
        vms = 0
        mms = 0
        cpu = 0
    return [name, pid, rss, vms, mms, cpu]


def getAllProcessInfo():
    """取出全部进程的进程名，进程ID，进程实际内存, 虚拟内存,CPU使用率
    """
    instances = []
    all_processes = list(psutil.process_iter())
    for proc in all_processes:
        proc.cpu_percent(interval=0)
    # 此处sleep1秒是取正确取出CPU使用率的重点
    time.sleep(1)
    for proc in all_processes:
        instances.append(getProcessInfo(proc))
    return instances


if __name__ == '__main__':
    # get_process_memory('python')
    #
    # test_p = multiprocessing.Process(target=test, args=(10,))
    #
    # test_p.start()
    # print("test_p PID:{}".format(test_p.pid))
    # print("Main PID:{}".format(os.getpid()))
    # ins = getAllProcessInfo()
    while True:
        logger.info('cpu(%) - {}'.format(psutil.cpu_percent(interval=1)))
    # test_m = multiprocessing.Process(target=get_pid_memory, args=(test_p.pid,))
    # test_m.start()
