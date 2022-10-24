import json
import logging
import pickle

import grpc

from blockchain.node.base_package import data_pb2, data_pb2_grpc

logger = logging.getLogger()


# 调用指定机器的远程方法，得到返回结果
# 此处选择运行不同的函数时请求的data和返回消息不同
# 运行uploadModel时，
# reuqest = {
#     'type': 0,
#     'name': '',
#     'file':None
# }
# 运行communicate时
# reuqest = {
#   optional: '',
#   status: '',
#   content:{}
# }
def runRemoteFunc(func='', data=None, HOST='127.0.0.1', PORT='8080'):
    if data is None:
        return 'Error, empty data!'
    # 监听频道
    conn = grpc.insecure_channel(HOST + ':' + PORT)
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
            logger.error('Cannot convert json string, error: {}'.format(e))
            response = {'type': -1, 'status': 500, 'content': 'ValueError:Cannot convert json string'}
    else:
        logger.error('Error, the function is not correct!')
        response = {'type': -1, 'status': 500, 'content': 'ValueError:Client error,the parameter is not correct!'}
    return response


def upload(data):
    mod = pickle.dumps(data['file'])
    actionrequest = data_pb2.actionrequest(type=data['type'], name=data['name'], file=mod)
    logger.info('actionrequest: {}'.format(actionrequest.type))
    return actionrequest


def communication(data):
    # 判断消息是否为str类型，若不是则转化为json字符串再发送
    if isinstance(data, str):
        request = data
    else:
        request = json.dumps(data)
    return request
