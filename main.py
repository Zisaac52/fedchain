import argparse
import logging

from blockchain.node.config import config
from blockchain.start import startNode

logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log',encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)  # 设置日志的级别
# fh.setFormatter(formatter)#设置的日志的输出
ch.setFormatter(formatter)
# logger.addHandler(fh) #logger对象可以添加多个fh和ch对象
logger.addHandler(ch)


def SetConfig(port, fsn=True):
    config['FirstNode'] = fsn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An argument inputs into command line')
    # param是参数的名字，type是要传入参数的数据类型，help是该参数的提示信息
    # 端口
    parser.add_argument('--port', required=True, type=str, default='', help='The port you want to listen')
    parser.add_argument('--nt', required=True, type=str, help='Node type (SN,EN,CN)')
    # 是否为0节点
    parser.add_argument('-z', action="store_true", help='It is a flag that this is the first node')
    # 获得传入的参数
    args = parser.parse_args()
    # 启动该节点
    startNode(attr=args.nt, port=args.port)
