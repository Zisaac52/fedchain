# 使用该方法返回正确的消息体，方便服务端解析
# 普通交流的结构
def Message(type, status, content):
    return {'type': type, 'status': status, 'content': content}


# 上传二进制流的结构
def FormData(type=0, name='', model_dict=None):
    if model_dict is None:
        return None
    return {'type': type, 'name': name, 'file': model_dict}
