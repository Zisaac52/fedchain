import logging
import multiprocessing
import sys

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, RegisterData
from blockchain.node.service.Server import serve
from blockchain.node.service.client import runRemoteFunc
from blockchain.node.vector_collect import get_endNode_perfomance

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
            self.AID = '{}:{}'.format(self.node_info.get('ip'), self.port)
            # 创建服务端
            self.server = multiprocessing.Process(target=serve, args=(self.node_info.get('ip'), self.port,))
        else:
            raise Exception("Error,the config 'port' is empty!")

    def register_self(self):
        # 根据自身节点的属性，判断是SN还是EN，SN则直接注册，EN则先选取一个节点计算状态，然后分配,先要收集自己的状态向量r
        self.en_handler()

    def en_handler(self):
        # 发送节点状态信息，获取待注册SN节点
        entry = config.get('entry_node')
        # 获取到向量,放入元组中
        tspeed = get_endNode_perfomance()
        vector = (0, 0, 0, tspeed,)
        comun = Message(type=3, status=200, content={'message': self.AID, 'data': vector})
        ok, resp = self.sender(comun, entry)
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, resp))
        # 向待注册SN发送注册信息，完成注册
        if ok:
            self.node_info['ip'] = config.get('publicIp')
            nodeinf = {'port': self.node_info['port'], 'ip': config.get('publicIp'), 'attr': self.node_info['attr']}
            reginfo = Message(type=0, status=200, content={'message': nodeinf})
            logger.debug('{} - {}' .format(sys._getframe().f_code.co_name, reginfo))
            self.sender(reginfo, resp.get('content').get('data'))
            # 给EN节点自身设置一个leader
            leader = Message(type=9, status=200, content=resp.get('content'))
            self.sender(leader, self.node_info)

    def sender(self, mesg, destination):
        """
        向服务器请求信息
        :param mesg: send to the server
        :param destination: {ip:' ' , port: ' '}
        :return: bool, response
        """
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, mesg))
        try:
            resp = runRemoteFunc(config['func']['sendMsg'], data=mesg, HOST=destination.get('ip'),
                                PORT=destination.get('port'))
            if resp.get('status') == 200:
                logger.info('{} - Successful, Info:{}'.format(sys._getframe().f_code.co_name, resp.get('content')))
                return True, resp
            else:
                raise Exception("Server error, response: {}".format(resp))
        except Exception as e:
            logger.error('{} - {} - {}'.format(self.__class__.__name__, sys._getframe().f_code.co_name, e))
            return False, None

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
