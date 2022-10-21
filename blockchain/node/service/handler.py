import json
import logging

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message
from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()


# 节点收到注册消息后执行该方法
def register_handler(message):
    msg = ''
    # 判断消息是否正确,存入配置中
    # 如果有错误，则返回错误提示，没有则不返回
    if config.get('node_attr').upper() != 'SN' or message is None:
        return Message(type=0, status=500, content={'message': 'Error, empty message or this is not a Server Node！'})
    if message.get('attr').upper() == 'SN':
        msg = check_and_set_node(message, 'SN')
        # 向其他SN节点广播列表
        message['type'] = 2
        for sn in config.get('node_list_sn'):
            resp = runRemoteFunc(config['func']['sendMsg'], data=message, HOST=sn.get('ip'),
                                PORT=sn.get('port'))
            if resp.get('status') != 200:
                logger.error('bordcast failure,node {}:{}'.format(sn.get('ip'), sn.get('port')))
    elif message.get('attr').upper() == 'EN':
        msg = check_and_set_node(message, 'EN')
    return Message(type=0, status=200, content={'message': msg})


def update_node_handler(message):
    msg = check_and_set_node(message, 'SN')
    logger.info("Node list updated,new SN {}".format(message))
    return Message(type=0, status=200, content={'message': msg})


# 判断当前是否重复注册
def check_and_set_node(message, attr):
    for n in config.get("node_list_{}".format(attr.lower())):
        if n.get('ip') + n.get('port') == message.get('ip') + message.get('port'):
            logger.error('This node is exist!')
            return 'The current {} node already exists, please do not repeat registration'.format(attr.lower())
    config.get("node_list_{}".format(attr.lower())).append(message)
    return 'successed！'


def bordcast_handler(message):
    pass


# 接收节点状态向量，计算
def calculate_status_vector_handler():
    pass
