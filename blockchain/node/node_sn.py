import logging
import multiprocessing

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, RegisterData
from blockchain.node.service.Server import serve
from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()


class NodeSN:

    def __init__(self, port=''):
        self.node_info = RegisterData()
        if port == '':
            self.port = self.node_info.get('port')
        else:
            self.port = port
        # 读取自己节点的属性
        if self.port != '' or self.port is not None:
            # 创建服务端
            self.server = multiprocessing.Process(target=serve, args=(self.node_info.get('ip'), self.port,))
        else:
            raise Exception("Error,the config 'port' is empty!")

    def register_self(self):
        # 自身是第一个节点，则不用注册了
        if config.get('FirstNode'):
            logger.warning('This is the first node, or node_list is not correct!')
            return
        self.sn_handler()

    def sn_handler(self):
        comun = Message(type=0, status=200, content=self.node_info)
        entry = config.get('entry_node')
        try:
            resp = runRemoteFunc(config['func']['sendMsg'], data=comun, HOST=entry.get('ip'),
                                        PORT=entry.get('port'))
            if resp.get('status') == 200:
                # config.get('node_list_sn').append(resp.get('content'))
                config.get('node_list_sn').append(entry)
                logger.info('Successful, Info:{}'.format(resp.get('content')))
            else:
                raise Exception("Server error, response: {}".format(resp))
        except Exception as e:
            logger.error(e)

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

