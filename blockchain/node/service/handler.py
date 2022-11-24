import json
import logging
import pickle
import random
import sys
import threading
from threading import Lock

import torch

from blockchain.node.config import config
from blockchain.node.entity.MessageEntity import Message, FormData, RegisterData
from blockchain.node.service.client import runRemoteFunc
from blockchain.node.service.handlerFL import start_fl_train_handler, calcdiff_handler
from blockchain.node.splitFL.SPclient import SPclient
from blockchain.node.splitFL.SPserver import SPserver

logger = logging.getLogger()


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwagrs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwagrs)
        return _instance[cls]

    return _singleton


# -----------------------------------------------------------------------------data
# 存储常用数据,如节点地址列表
@Singleton
class Handler(object):
    # 在SN下的属性
    SN_NODE_LIST = list()
    EN_NODE_LIST = list()
    # 在EN下的属性
    EN_leader = None
    # 公共属性
    SPSERVICE = None
    # 保存状态向量
    STVECTOR = None
    # 服务端模型，只有SN有
    SEVERMODEL = None
    # 锁
    lock = Lock()

    def __init__(self):

        # 创建学习服务,若是SN节点则使用服务端,EN节点则用客户端
        if config.get('node_attr').upper() == 'SN':
            self.SPSERVICE = SPserver()
            torch.save(self.SPSERVICE.model_sev.state_dict(), './data/sp/server-0.pth')
        else:
            self.SPSERVICE = SPclient()
        # 如果不是第一个节点，则将入口节点加入列表
        if not config.get('FirstNode') and len(self.SN_NODE_LIST) == 0:
            self.SN_NODE_LIST.append(config.get('entry_node'))
            logger.info('{} - New list {}'.format(sys._getframe().f_code.co_name, self.SN_NODE_LIST))
        else:
            self.STVECTOR = {}


# -----------------------------------------------------------------------------handler
# 用于对服务端消息处理，
# 增加handler时需要向Server类注册
def register_handler(message):
    """
    节点收到注册消息后执行该方法--0
    判断消息是否正确,存入配置中
    如果有错误，则返回错误提示，没有则不返回
    :param message: content ->dict{'message':msg}
    :return:
    """
    nt = message.get('message').get('attr').upper()
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, nt))
    if config.get('node_attr').upper() != 'SN' or message is None:
        return Message(type=0, status=500, content={'message': 'Error, empty message or this is not a Server Node！'})
    if nt == 'SN':
        ok, msg = check_and_set_node(message, 'SN')
        # 验证正确则向其他节点广播
        if ok:
            # 向其他SN节点发送列表
            bordcast({'message': Handler().SN_NODE_LIST}, 2, 'SN')
            logger.info('{} - SN_NODE_LIST:{}'.format(sys._getframe().f_code.co_name, Handler().SN_NODE_LIST))
            # return Message(type=2, status=200, content={'message': config.get('node_list_sn'), 'nt': nt})
        else:
            return Message(type=0, status=500, content={'message': msg})
    elif nt == 'EN':
        oken, enmsg = check_and_set_node(message, 'EN')
        if oken:
            logger.info('{} - EN_NODE_LIST:{}'.format(sys._getframe().f_code.co_name, Handler().EN_NODE_LIST))
        return Message(type=0, status=200, content={'message': enmsg})
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
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, message.get('message')))
    return set_node_list(nl)


def calculate_status_vector_handler(message):
    """
    接收节点状态向量，计算，如果节点状态向量满足SN中任一节点，则将返回该节点信息--3
    从该方法开始，所有传送数据json中使用data，结构为 content-> data
    :return:
    """
    vector = message.get('data')
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, vector))
    if len(Handler().SN_NODE_LIST) > 0:
        # 记录下状态向量
        Handler().STVECTOR[message.get('message')] = vector
        # 随机分配管理节点
        nc = random.randint(0, len(Handler().SN_NODE_LIST) - 1)
        return Message(type=3, status=200, content={"message": 'Calculate vector successfully！',
                                                    "data": Handler().SN_NODE_LIST[nc]})
    else:
        return Message(type=3, status=500, content={'message': 'No SN node!'})


def send_task_handler(message):
    """
    将学习完成的任务发送出去
    :param message: 传入的整个actionrequest
    :return: dfx: 服务器计算梯度结果，返回给客户端
            message: 梯度附带信息
    """
    with Handler().lock:
        # 解码文件
        state_dict = pickle.loads(message.file)
        # 取出里面的信息
        apdmsg = json.loads(message.message)
        # 取出里面的targets
        targets = state_dict.get('targets')
        dfx = Handler().SPSERVICE.train(state_dict.get('dfx'), targets, apdmsg.get('flag'), apdmsg.get('epoch'))
    logger.debug('{} - training...'.format(sys._getframe().f_code.co_name))
    return dfx, {'message': 'send_task_handler'}


def distribute_task_handler(message):
    """
    接收后台命令，分发任务，对应type --6 \n
    消息组成：{task: '', data: {}, tag1: '', tag2: ''....}\n
    任务类型-> 训练T，停止P，保存模型S，评估模型E \n
    option = {\n
        # 训练模型\n
        'train()': 'T',\n
        # 暂停\n
        'pause()': 'P',\n
        # 保存\n
        'save()': 'S',\n
        # 评估模型\n
        'eval()': 'E',\n
        # 获取SN列表\n
        'getSN()': 'GSN',\n
        # 下载服务端模型\n
        'getModel()': 'DL'\n
    }\n
    :param message: dict->由客户端请求的发送的内容 content
    :return: Message()
    """
    cmd = message.get('task')
    if cmd == 'T':
        # 接收到命令后，向各SN节点发送开始命令，由SN发布训练任务到EN，EN节点开始训练
        msg = {'message': 'train', 'cmd': cmd}
        res = bordcast(msg, 7, 'SN')
        if res is None:
            return Message(type=6, status=500, content={"message": 'No SN node!'})
        return Message(type=6, status=200, content={"message": 'success', 'data': res})
    elif cmd == 'GSN':
        return Message(type=6, status=200, content={"message": 'success', 'data': Handler().SN_NODE_LIST})
    elif cmd == 'DL':
        return Message(type=6, status=200, content={"message": 'download'})
    elif cmd == 'SV':
        return Message(type=6, status=200, content={'message': 'success', 'data': Handler().STVECTOR})
    elif cmd == 'FLT':
        msg = {'message': 'train', 'cmd': cmd, 'epoch': 20}
        res = bordcast(msg, 20, 'SN')
        return Message(type=6, status=200, content={"message": res})
    else:
        return Message(type=-1, status=500, content={"message": 'error'})


def start_self_en_task_handler(message):
    """
    接收到来自主节点的命令，判断并下发给EN，开始训练 --7
    :return:
    """
    msg = message.get('content')
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, msg))
    bordcast(message=msg, mytype=8, nodeattr='EN')
    pass


def get_SN_train_signal_handler(message):
    """
    接收SN信号，开始训练 --8
    :return:
    """
    logger.info('{} - start training!'.format(sys._getframe().f_code.co_name))
    en_train_handler()
    return Message(type=8, status=200, content={"message": 'success'})


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


def test_network_handler(message):
    """
    测试网络用 --10
    :return:
    """
    if config.get('node_attr').upper() == 'SN':
        return Message(type=10, status=200, content={"message": 'success'})
    else:
        return Message(type=10, status=500, content={"message": 'refused'})


def test_fl_handler(message):
    """
    加载fl的训练网络进行训练 --20
    :param message: content -> dict{}
    :return: msg
    """
    msg = {'node': Handler().EN_NODE_LIST[0], 'epoch': 20}
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, message))
    if config.get('node_attr').upper() == 'SN':
        res = start_fl_train_handler(msg)
        return res


# -----------------------------------------------------------------------------EN_Handler
# 用于处理EN节点相关事务
def en_train_handler(message=None):
    """
    EN节点训练的处理方法
    供客户端调用
    :return:
    """
    if config.get('node_attr').upper() == 'EN':
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, config.get('node_attr').upper()))
        Handler().SPSERVICE.train(upload_remote_dict)
    pass


def set_en_leader_handler(message):
    """
    EN设置注册节点 --9
    :param message: content -> dict()   {'port': '8081', 'ip': '10.0.12.16', 'publicIp': puip, 'attr': 'SN'}
    :param node:
    :return:
    """
    nodes = message.get('data')
    nodes['ip'] = nodes['publicIp']
    Handler().EN_leader = nodes
    message.get()
    logger.info('{} - {}'.format(sys._getframe().f_code.co_name, Handler().EN_leader))


def upload_remote_dict(dfx, targets, flag, epoch):
    """
    根据传入的参数运行远程方法获得反向梯度
    :param flag: 当前是否完成了一轮
    :param epoch:
    :param dfx:
    :param targets:
    :return:
    """
    if Handler().EN_leader is None:
        raise ValueError('No en leader!')
    else:
        msg = FormData(type=1, name='mnist', message={'message': 'mnist_Net', 'flag': flag, 'epoch': epoch},
                    model_dict={'dfx': dfx, 'targets': targets})
        resp = runRemoteFunc(config['func']['upload'], data=msg, HOST=Handler().EN_leader.get('ip'),
                            PORT=Handler().EN_leader.get('port'))
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, resp.message))
        dfx = pickle.loads(resp.file)
    return dfx


def get_fl_diff_handler(message):
    """
    获取到客户端的diff，传给服务端聚合
    :param message: actionrequest -> type=10, name='mnist', message={'message': 'mnist_Net', 'version': ver}, file
    :return:
    """
    f = pickle.loads(message.file)
    msg = json.loads(message.message)
    diff, msgs = calcdiff_handler(f, msg.get('version'))
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, msgs))
    return diff, msgs


# -----------------------------------------------------------------------------utils
# 常规工具方法，用于提供handler调用，
# 不参与直接通信处理
#
def bordcast(message, mytype, nodeattr):
    """
    处理需要广播的业务
    :param nodeattr: 当前节点属性 SN or EN
    :param message: 传入的部分为content->dict
    :param mytype:
    :return: error_list
    """
    nodelist = Handler().SN_NODE_LIST if nodeattr.upper() == 'SN' else Handler().EN_NODE_LIST
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, nodelist))
    error_list = []
    if len(nodelist) > 0:
        bordcastMsg = Message(type=mytype, status=200, content=message)
        for sn in nodelist:
            # 多线程发送，防止阻塞
            def run():
                resp = runRemoteFunc(config['func']['sendMsg'], data=bordcastMsg, HOST=sn.get('ip'),
                                     PORT=sn.get('port'))
                if resp.get('status') != 200:
                    logger.error(
                        'bordcast failure,node {}:{}, something wrong about->{}'.format(sn.get('ip'), sn.get('port'),
                                                                                        resp))
                    error_list.append({'ip': sn.get('ip'), 'port': sn.get('port'), 'response': resp})
                else:
                    logger.info('{} - {}'.format(sys._getframe().f_code.co_name, resp))

            t = threading.Thread(target=run)
            t.setDaemon(True)  # 把子进程设置为守护线程，必须在start()之前设置
            t.start()
        return error_list
    else:
        logger.warning('Empty SN list!')
        return None


def check_and_set_node(message, attr):
    """
    判断当前是否重复注册
    :param message: content -> dict{'message': {'port': '8082', 'ip': '222.197.211.114', 'attr': 'EN'}}
    :param attr: SN or EN
    :return: bool,string
    """
    msg = message.get('message')
    nlt = Handler().SN_NODE_LIST if attr.upper() == 'SN' else Handler().EN_NODE_LIST
    if len(nlt) != 0:
        # 判断节点是否重复
        for n in nlt:
            if '{}:{}'.format(n.get('ip'), n.get('port')) == '{}:{}'.format(msg.get('ip'), msg.get('port')):
                return False, 'The current {} node already exists, please do not repeat registration'.format(
                    attr.lower())
    logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, msg))
    nlt.append(msg)
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
        logger.debug('{} - {}'.format(sys._getframe().f_code.co_name, nlist))
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
