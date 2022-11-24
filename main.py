import argparse
import json
import logging
import socket
import time

import torch

from blockchain.node.splitFL.SPeval import evalmodel
from blockchain.node.splitFL.splitmodel import mnist_Net_client, mnist_Net_server

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


def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def SetConfig(port='', attr='', entry='', ip='',fsn=True):
    """
    保存用户输入的配置\n
    :param ip:
    :param entry:
    :param attr: 节点属性
    :param port: 端口
    :param fsn: 是否为第一个节点
    :return:
    """
    with open('nodeconfig.json', 'r') as f:
        conf = json.load(f)
        conf['FirstNode'] = fsn
        conf['ip'] = get_host_ip()
        conf['publicIp'] = ip if ip != '' else get_host_ip()
        if attr != '':
            conf['node_attr'] = attr
        if port != '':
            conf['port'] = port
        if entry != '':
            conf['entry_node']['ip'] = entry.split(':')[0]
            conf['entry_node']['port'] = entry.split(':')[1]
    json.dump(conf, open('nodeconfig.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An argument inputs into command line')
    # param是参数的名字，type是要传入参数的数据类型，help是该参数的提示信息
    # 端口
    parser.add_argument('--ip', type=str, default='', help='The ip you self')
    parser.add_argument('--port', required=True, type=str, default='', help='The port you want to listen')
    parser.add_argument('--nt', required=True, type=str, help='Node type (SN,EN,CN)')
    parser.add_argument('--entry', required=True, type=str, help='eg. 127.0.0.1:8080')
    # 是否为0节点
    parser.add_argument('-z', action="store_true", help='It is a flag that this is the first node')
    # 获得传入的参数
    args = parser.parse_args()
    SetConfig(args.port, args.nt, args.entry, args.ip, args.z)
    # with open('nodeconfig.json', 'r') as f:
    #     conf = json.load(f)
    #     print(conf)
    from blockchain.start import startNode
    # # 启动该节点
    startNode(attr=args.nt, port=args.port)
    # serve('222.197.211.122', '8080')
    # with open('nodeconfig.json', 'w+') as f:
    #     json.dump(config_sn, f)
