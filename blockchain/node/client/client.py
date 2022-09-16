import grpc

from blockchain.node.base_package import data_pb2, data_pb2_grpc


# {
#   optional: '',
#   status: '',
#   content:{}
# }
# 调用指定机器的远程方法，得到返回结果
def runRemoteFunc(data='', HOST='127.0.0.1',PORT='8080'):
    # 监听频道
    conn = grpc.insecure_channel(HOST + ':' + PORT)
    # 客户端使用Stub类发送请求,参数为频道,为了绑定链接
    client = data_pb2_grpc.FormatDataStub(channel=conn)
    # 返回的结果就是proto中定义的类
    response = client.DoFormat(data_pb2.actionrequest(text=data))
    return response
