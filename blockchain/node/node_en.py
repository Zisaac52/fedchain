import logging
import multiprocessing

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, RegisterData
from blockchain.node.service.Server import serve
from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()


class NodeEN:

    def __init__(self, port=None):
        self.node_info = RegisterData()
        if port is None:
            self.port = self.node_info.get('port')
        else:
            self.port = port
        # 读取自己节点的属性
        if self.port != '' or self.port is not None:
            # 创建服务端
            self.server = multiprocessing.Process(target=serve, args=('127.0.0.1', self.port,))
        else:
            raise Exception("Error,the config 'port' is empty!")

    def register_self(self):
        # 根据自身节点的属性，判断是SN还是EN，SN则直接注册，EN则先选取一个节点计算状态，然后分配
        pass

    def en_handler(self):
        # 发送节点状态信息，获取待注册SN节点

        # 向待注册SN发送注册信息，完成注册
        pass

    # {
    #   optional: '',
    #   status: '',
    #   content:{}
    # }
    # 加载配置文件
    # 节点开启监听服务
    # 判断自己的状态，向服务器SN注册自己
    def startNode(self):
        self.server.start()
        self.register_self()
        # 初始化，判断自身是否是第一个节点，否则寻找配置的节点源进行加入
