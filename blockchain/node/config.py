import json
import os.path


def loadjson():
    """
    加载json配置
    :return: dict
    """
    if os.path.exists('./nodeconfig.json'):
        with open('./nodeconfig.json', 'r') as f:
            conf = json.load(f)
    else:
        raise Exception('There is no config file!')
    return conf


# 服务端节点启动时加载的配置
config = loadjson()

# # 终端节点启动时加载的配置
# config_en = {
#     'node_attr': 'EN',
#     'port': '8080',
#     'ip': get_host_ip()
# }
#
# config_sn = {
#     # grpc配置的函数
#     'func': {
#         'upload': 'uploadModel',
#         'sendMsg': 'communicate'
#     },
#     # 节点列表
#     # {
#     #   ip：127.0.0.1,
#     #   port：8080,
#     #   node_attr：
#     # }
#     # 当前是第一个节点
#     'FirstNode': True,
#     # {'ip': '124.71.0.131', 'port': '8080', 'attr': 'SN'}
#     # 配置入口SN节点
#     'entry_node': {'ip': '222.197.211.116', 'port': '8080', 'attr': 'SN'},
#     # 当前任务类型 0-注册节点，1-
#     'type': [0, 1, 2, 3, 4],
#     # 该节点的属性，（SN,EN,CN）=>(服务节点，终端，外部接入节点)
#     # 只有SN节点会被广播到其他SN节点进行服务发现，EN负责联系某个节点接收训练任务，
#     # CN只作为交流节点使用，不进行训练任务，也不参与共识，但可以获取当前模型训练状态和网络信息
#     'node_attr': 'SN',
#     'port': '8080',
#     'ip': get_host_ip()
# }
