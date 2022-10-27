import json
import logging
import sys

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message
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
class Handler(object):
    SN_NODE_LIST = list()
    EN_NODE_LIST = list()

    def __init__(self):
        # 如果不是第一个节点，则将入口节点加入列表
        if not config.get('FirstNode') and len(self.SN_NODE_LIST) == 0:
            self.SN_NODE_LIST.append(config.get('entry_node'))
            logger.info('{} - New list {}'.format(sys._getframe().f_code.co_name, self.SN_NODE_LIST))


def register_handler(message):
    """
    节点收到注册消息后执行该方法--0
    判断消息是否正确,存入配置中
    如果有错误，则返回错误提示，没有则不返回
    :param message:
    :return:
    """
    nt = message.get('attr').upper()
    if config.get('node_attr').upper() != 'SN' or message is None:
        return Message(type=0, status=500, content={'message': 'Error, empty message or this is not a Server Node！'})
    if nt == 'SN':
        ok, msg = check_and_set_node(message, 'SN')
        # 验证正确则向其他节点广播
        if ok:
            # 向其他SN节点发送列表
            bordcast(Handler().SN_NODE_LIST, 2)
            logger.info('{} - {}'.format(sys._getframe().f_code.co_name, Handler().SN_NODE_LIST))
            # return Message(type=2, status=200, content={'message': config.get('node_list_sn'), 'nt': nt})
        else:
            return Message(type=0, status=500, content={'message': msg})
    elif nt == 'EN':
        return Message(type=0, status=200, content={'message': check_and_set_node(message, 'EN')})
    else:
        return Message(type=-1, status=500, content={'message': 'parameter is incorrect！'})


def networkinfo_handler(message):
    """
    处理网络相关的信息，返回给请求用户--1
    :param message:
    :return:
    """
    pass


def update_node_handler(message):
    """
    更新SN节点列表--2
    :param message:
    :return: dict
    """
    nl = message.get('message')
    return set_node_list(nl)


def calculate_status_vector_handler():
    """
    接收节点状态向量，计算，如果节点状态向量满足SN中任一节点，则将返回该节点信息--3
    :return:
    """
    pass


def success_handler(message):
    """
    发送成功信息的处理--4
    :param message:
    :return:
    """
    logger.info(message)


def error_handler():
    """
    对于参数错误时的处理 --5
    :return:
    """
    pass


def bordcast(message, mytype):
    """
    处理需要广播的业务
    :param message: dict{type=0, status=200, content={'message': message}}
    :param mytype:
    :return:
    """
    if len(Handler().SN_NODE_LIST) > 0:
        bordcastMsg = Message(type=mytype, status=200, content={'message': message})
        for sn in Handler().SN_NODE_LIST:
            resp = runRemoteFunc(config['func']['sendMsg'], data=bordcastMsg, HOST=sn.get('ip'),
                                 PORT=sn.get('port'))
            if resp.get('status') != 200:
                logger.error(
                    'bordcast failure,node {}:{}, something wrong about->{}'.format(sn.get('ip'), sn.get('port'), resp))
            else:
                logger.info('{} - {}'.format(sys._getframe().f_code.co_name, resp))
    else:
        logger.warning('Empty SN list!')


def check_and_set_node(message, attr):
    """
    判断当前是否重复注册
    :param message: dict{type=0, status=200, content={'message': message}}
    :param attr: SN or EN
    :return: bool,string
    """
    nlt = Handler().SN_NODE_LIST if attr.upper() == 'SN' else Handler().EN_NODE_LIST
    if len(nlt) != 0:
        # 判断节点是否重复
        for n in nlt:
            if n.get('ip') + n.get('port') == message.get('ip') + message.get('port'):
                return False, 'The current {} node already exists, please do not repeat registration'.format(
                    attr.lower())
    nlt.append(message)
    return True, 'Add new node successfully!'


def set_node_list(nlist):
    """
    接收到节点广播的列表后去重，加入自己的节点列表
    :param nlist:
    :return:
    """
    if len(nlist) > 0:
        n_self = []
        n_other = []
        for n in Handler().SN_NODE_LIST:
            n_self.append('{}:{}'.format(n.get('ip'), n.get('port')))
        # logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, Handler().SN_NODE_LIST))
        for mn in nlist:
            # 排除自身地址
            if '{}:{}'.format(mn.get('ip'), mn.get('port')) != '{}:{}'.format(config.get('ip'), config.get('port')):
                if mn.get('attr').upper() == 'SN':
                    n_other.append('{}:{}'.format(mn.get('ip'), mn.get('port')))
        cln_otr = list(set(n_self) ^ set(n_other))
        # logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, nlist))
        for host in cln_otr:
            # 排除入口地址
            if '{}:{}'.format(config.get('entry_node').get('ip'), config.get('entry_node').get('port')) != host:
                hs = host.split(':')
                Handler().SN_NODE_LIST.append({'ip': hs[0], 'port': hs[1], 'attr': 'SN'})
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, Handler().SN_NODE_LIST))
        return Message(type=4, status=200, content={'message': 'success!'})
    else:
        return Message(type=4, status=200, content={'message': 'Empty list!'})
