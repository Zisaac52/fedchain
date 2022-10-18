import logging
import multiprocessing

import torch
import torchvision.models

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, FormData
from blockchain.node.service.Server import serve
from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log',encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)# 设置日志的级别
# fh.setFormatter(formatter)#设置的日志的输出
ch.setFormatter(formatter)
# logger.addHandler(fh) #logger对象可以添加多个fh和ch对象
logger.addHandler(ch)
# {
#   optional: '',
#   status: '',
#   content:{}
# }
# 加载配置文件
# 节点开启监听服务
def startNode():
    multiprocessing.Process(target=serve, args=('127.0.0.1', '8080',)).start()
    comun = Message(type=0, status=200, content={'server': '127.0.0.1', 'port': '8080'})
    resp = runRemoteFunc(config['func']['sendMsg'], data=comun)
    logger.info('消息测试：{}'.format(resp))
    models = torchvision.models.mnasnet1_0()
    request = FormData(type=0,name='mnasnet1_0', model_dict=models.state_dict())
    resp = runRemoteFunc(config['func']['upload'], data=request)
    logger.info(resp)

    # for i in range(10):
    #     runRemoteFunc(data=jstr)
    pass


if __name__ == '__main__':
    startNode()
    pass
