import json

from blockchain.node.client.client import runRemoteFunc
from blockchain.node.server.Server import serve
import multiprocessing


# {
#   optional: '',
#   status: '',
#   content:{}
# }
# 加载配置文件
# 节点开启监听服务
def startNode():
    multiprocessing.Process(target=serve, args=('127.0.0.1', '8080',)).start()
    dt = dict(optional='SAVE', status='200', content=dict(name='yang', time='2022-9-5 15:53'))
    jstr = str(json.dumps(dt))
    resp = runRemoteFunc(data=jstr)
    print(resp)
    # for i in range(10):
    #     runRemoteFunc(data=jstr)
    pass


if __name__ == '__main__':
    startNode()
    pass
