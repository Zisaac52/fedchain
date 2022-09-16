import hashlib
import json

import grpc
import time
from concurrent import futures

from blockchain.node.base_package import data_pb2, data_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# {
#   optional: '',
#   status: '',
#   content:{}
# }

# 调用数据库存储区块
def save(data):
    result = hashlib.sha256(str(data).encode()).hexdigest()
    data['preHash'] = result
    data['selfHash'] = result
    print('create block:', data)
    pass


def parseData(jsonData):
    data = None
    print(jsonData)
    try:
        data = json.loads(jsonData)
    except:
        print('解析json失败')
    if data is not None or data['optional'] is not None:
        switch(data['optional'], data)


def switch(op, data):
    if op == 'SAVE':
        save(data['content'])
    elif op == 'TEST':
        print('Test...')
    pass


def serve(HOST='localhost', PORT='8081'):
    # 定义服务器并设置最大连接数,corcurrent.futures是一个并发库，类似于线程池的概念
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))  # 创建一个服务器
    data_pb2_grpc.add_FormatDataServicer_to_server(FormatData(), grpcServer)  # 在服务器中添加派生的接口服务（自己实现了处理函数）
    grpcServer.add_insecure_port(HOST + ':' + PORT)  # 添加监听端口
    grpcServer.start()  # 启动服务器
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)  # 关闭服务器


# 通过rpc调用的函数
class FormatData(data_pb2_grpc.FormatDataServicer):
    # 重写接口函数
    def DoFormat(self, request, context):
        # 节点发来消息，对消息进行解析分类处理，并返回处理结果
        strs = request.text
        parseData(strs)
        return data_pb2.actionresponse(text='the result was not valid!')  # 返回一个类实例
