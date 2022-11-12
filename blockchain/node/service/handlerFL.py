# 传统联邦学习handler
import logging
import pickle
import sys

import torch

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import FormData
from blockchain.node.fed.client import Client
from blockchain.node.fed.server import Server
from blockchain.node.service.client import runRemoteFunc
logger = logging.getLogger()


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwagrs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwagrs)
        return _instance[cls]

    return _singleton


@Singleton
class FLHandler:
    gobal_model = None
    client_model = None
    SERVICE = None
    CLIENT= None

    def __init__(self):
        if config.get('node_attr').upper() == 'SN':
            self.SERVICE = Server()

    def setClient(self, client):
        self.CLIENT = client


# -----------------------------client
def calcdiff_handler(file, ver):
    """
    调用client计算，返回diff
    :param ver: 当前模型版本
    :param file: 模型
    :param message: message-> {'type': type, 'name': name, 'message': message, 'file': model_dict}
    :return: diff->tensor
    """
    logger.debug('{} - model version：{}'.format(sys._getframe().f_code.co_name, ver))
    if config.get('node_attr').upper() == 'EN':
        if FLHandler().CLIENT is None:
            FLHandler().setClient(Client(file, ver))
        logger.debug('{} - CLIENT is None:{}'.format(sys._getframe().f_code.co_name, FLHandler().CLIENT is None))
        FLHandler().CLIENT.setModelFromServer(file, ver)
        diff = FLHandler().CLIENT.local_train()
        return diff


# -----------------------------server

def start_fl_train_handler(message):
    """
    接收控制台命令，开始训练
    :param message: message-> dict{...content:{epoch: 10}}
    :return: 没想好
    """
    for i in range(message.get('epoch')):
        diffx = load_client_diff(message.get('node'))
        FLHandler().SERVICE.start_train(diff=diffx)
        model, _ = FLHandler().SERVICE.get_model()
        torch.save(model.state_dict(), 'data/flm/gmodel-{}.pth'.format(i))
        logger.info('{} - model save:{}'.format(sys._getframe().f_code.co_name, 'data/flm/gmodel-{}.pth'.format(i)))
    return 'finish training'


# -----------------------------utils
def load_client_diff(node):
    logger.debug('{} - diff get:{}'.format(sys._getframe().f_code.co_name, node))
    model, ver = FLHandler().SERVICE.get_model()
    msg = FormData(type=10, name='mnist', message={'message': 'mnist_Net', 'version': ver}, model_dict=model)
    # actionresponse(type=1, name='uploadModel', message=json.dumps(msg), file=resps)
    resp = runRemoteFunc(config['func']['upload'], data=msg, HOST=node.get('ip'), PORT=node.get('port'))
    diff = pickle.loads(resp.file)
    return diff
