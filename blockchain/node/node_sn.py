import logging
import multiprocessing

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, RegisterData
from blockchain.node.service.Server import serve
from blockchain.node.service.client import runRemoteFunc

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


class NodeSN:

    def __init__(self):
        self.node_info = RegisterData()
        # 读取自己节点的属性
        if self.node_info.get('port') is not None:
            # 创建服务端
            self.server = multiprocessing.Process(target=serve, args=('127.0.0.1', self.node_info.get('port'),))
        else:
            raise Exception("Error,the config 'port' is empty!")

    def register_self(self):
        # 根据自身节点的属性，判断是SN还是EN，SN则直接注册，EN则先选取一个节点计算状态，然后分配
        if self.node_info.get('attr').upper() == 'SN':
            self.sn_handler()
        else:
            pass
        if len(config.get('node_list_sn')) is 0:
            logger.warning('This is the first node, or node_list is not correct!')
            return

    def sn_handler(self):
        comun = Message(type=0, status=200, content=self.node_info)
        origin_node = config.get('node_list_sn')
        for node in origin_node:
            if node.get('attr').upper() == 'SN':
                try:
                    resp = runRemoteFunc(config['func']['sendMsg'], data=comun, HOST=node.get('ip'),
                                        PORT=node.get('port'))
                    if resp.get('status') == 200:
                        origin_node.append(resp.get('content'))
                        logger.info('Successful registration of service nodes！')
                    else:
                        raise Exception("Server error!")
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


if __name__ == '__main__':
    NodeSN().startNode()
    pass
