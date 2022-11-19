import json
import logging
import pickle
import sys

import grpc

from blockchain.node.base_package.proto import data_pb2_grpc, data_pb2
from blockchain.node.service.JsonEncoder import SetEncoder

logger = logging.getLogger()


def runRemoteFunc(func='', data=None, HOST='127.0.0.1', PORT='8080'):
    """
    调用指定机器的远程方法，得到返回结果
    此处选择运行不同的函数时请求的data和返回消息不同
    运行uploadModel时，\n
    reuqest = {
        'type': 0,
        'name': '',
        'file':None
    }\n
    运行communicate时\n
    reuqest = {
        type: '',
        status: '',
        content:{}
    }\n
    :param func:
    :param data:
    :param HOST:
    :param PORT:
    :return:
    """
    if data is None:
        return 'Error, empty data!'
    # 监听频道
    conn = grpc.insecure_channel('{}:{}'.format(HOST, PORT))
    # 客户端使用Stub类发送请求,参数为频道,为了绑定链接
    client = data_pb2_grpc.FormDataStub(channel=conn)
    # 判断并运行相应的方法
    if func.lower() == 'uploadModel'.lower():
        actionreuqest = upload(data)
        response = client.uploadModel(actionreuqest)
    elif func.lower() == 'communicate'.lower():
        # 调用远程communicate方法并取得返回值,返回值是一个字符串，需要转换为dict
        response = client.communicate(data_pb2.request(message=communication(data))).message
        try:
            response = json.loads(response)
        except ValueError as e:
            logger.error('{} - Cannot convert json string, error: {}'.format(sys._getframe().f_code.co_name, e))
            response = {'type': -1, 'status': 500, 'content': {'message': 'ValueError:Cannot convert json string'}}
    else:
        logger.error('{} - Error, the function is not correct!'.format(sys._getframe().f_code.co_name))
        response = {'type': -1, 'status': 500, 'content': {
            'message': 'ValueError:Client error,the parameter is not correct!'}}
    return response


def upload(data):
    try:
        mod = pickle.dumps(data['file'])
        msg = json.dumps(data['message'], cls=SetEncoder)
        actionrequest = data_pb2.actionrequest(type=data['type'], name=data['name'], message=msg, file=mod)
        # logger.info('{} - actionrequest: {}'.format(sys._getframe().f_code.co_name, actionrequest.type))
    except Exception as e:
        logger.error('{} - ClientError:{}'.format(sys._getframe().f_code.co_name, e))
        actionrequest = data_pb2.actionrequest(type=-1, name='error', message='{}', file=None)
    return actionrequest


def communication(data):
    # 判断消息是否为str类型，若不是则转化为json字符串再发送
    if isinstance(data, str):
        request = data
    else:
        request = json.dumps(data, cls=SetEncoder)
    return request
