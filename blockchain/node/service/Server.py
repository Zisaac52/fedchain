import hashlib
import json
import pickle
import logging
import grpc
import time
from concurrent import futures

from blockchain.node.base_package import data_pb2, data_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 100*1024*1024
logger = logging.getLogger()


def serve(HOST='localhost', PORT='8081'):
    # 定义服务器并设置最大连接数,corcurrent.futures是一个并发库，类似于线程池的概念
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])  # 创建一个服务器
    data_pb2_grpc.add_FormDataServicer_to_server(FormData(), grpcServer)  # 在服务器中添加派生的接口服务（自己实现了处理函数）
    grpcServer.add_insecure_port(HOST + ':' + PORT)  # 添加监听端口
    logger.info('server start, listen in-{}'.format(PORT))
    grpcServer.start()  # 启动服务器
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)  # 关闭服务器


# 通过rpc调用的函数
class FormData(data_pb2_grpc.FormDataServicer):
    # 重写接口函数

    # 接收传入的文件bytes，用于向全局模型聚合
    # 返回接收成功的信息
    def uploadModel(self, file_request, context):
        state_dict = pickle.loads(file_request.file)
        return data_pb2.actionresponse(message='uploadModel')

    # 普通的服务器节点之间交流
    def communicate(self, request, context):
        logger.info('Message：{}'.format(request))
        return data_pb2.response(message='communicate')
