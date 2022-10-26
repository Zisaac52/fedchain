import json
import logging

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message
from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()


# 节点收到注册消息后执行该方法--0
def register_handler(message):
    msg = ''
    # 判断消息是否正确,存入配置中
    # 如果有错误，则返回错误提示，没有则不返回
    if config.get('node_attr').upper() != 'SN' or message is None:
        return Message(type=0, status=500, content={'message': 'Error, empty message or this is not a Server Node！'})
    if message.get('attr').upper() == 'SN':
        ok, msg = check_and_set_node(message, 'SN')
        # 验证正确则向其他节点广播
        if ok:
            # 向其他SN节点发送列表
            bordcast_handler(config.get('node_list_sn'), 2)
            logger.info(config.get('node_list_sn'))
            return Message(type=0, status=200, content={'message': config.get('node_list_sn')})
        else:
            return Message(type=0, status=500, content={'message': msg})
    elif message.get('attr').upper() == 'EN':
        return Message(type=0, status=200, content={'message': check_and_set_node(message, 'EN')})


# 处理网络相关的信息，返回给请求用户--1
def networkinfo_handler(message):
    pass


# --2
def update_node_handler(message):
    msg = 'update successfully!'
    logger.info("Node list updated,new SN {}".format(message))
    return Message(type=2, status=200, content={'message': msg})


# 接收节点状态向量，计算--3
def calculate_status_vector_handler():
    pass


# 处理需要广播的业务
def bordcast_handler(message, mytype):
    if len(config.get('node_list_sn')) > 0:
        bordcastMsg = Message(type=mytype, status=200, content={'message': message})
        for sn in config.get('node_list_sn'):
            resp = runRemoteFunc(config['func']['sendMsg'], data=bordcastMsg, HOST=sn.get('ip'),
                                PORT=sn.get('port'))
            if resp.get('status') != 200:
                logger.error('bordcast failure,node {}:{}'.format(sn.get('ip'), sn.get('port')))
            else:
                logger.info(resp)
    else:
        logger.warning('Empty SN list!')


# 判断当前是否重复注册
def check_and_set_node(message, attr):
    if len(config.get("node_list_{}".format(attr.lower()))) != 0:
        # 判断节点是否重复
        for n in config.get("node_list_{}".format(attr.lower())):
            if n.get('ip') + n.get('port') == message.get('ip') + message.get('port'):
                return False, 'The current {} node already exists, please do not repeat registration'.format(
                    attr.lower())
    config.get("node_list_{}".format(attr.lower())).append(message)
    return True, 'Add new node successfully!'


def runRemoteFuncTest(fun, data, HOST, PORT):
    logger.info('运行函数：{}，发送参数：{}，主机地址：{}:{}'.format(fun, data, HOST, PORT))
    return {'status': 200}
    pass

