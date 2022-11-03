# 使用该方法返回正确的消息体，方便服务端解析
# 普通交流的结构
from blockchain.node.config import config


def Message(type, status, content):
    return {'type': type, 'status': status, 'content': content}


# 上传二进制流的结构
def FormData(type=0, name='', message='', model_dict=None):
    if model_dict is None:
        return None
    return {'type': type, 'name': name, 'message': message, 'file': model_dict}


# 封装注册节点消息体
def RegisterData():
    attr = config.get('node_attr')
    port = config.get('port')
    ip = config.get('ip')
    return {'port': port, 'ip': ip, 'attr': attr}
