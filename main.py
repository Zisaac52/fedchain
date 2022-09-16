import config
from fl.client import Client
from fl.server import Server

if __name__ == '__main__':
    # 创建一个服务
    sr = Server()
    # 加入工作节点
    for i in range(config.my_conf['client.amount']):
        # 注册客户端，向每个节点分发模型
        sr.addClient(Client(lr=config.my_conf['learn_rate']), '127.0.0.1:808{}'.format(i))
    # 开始迭代训练
    for i in range(config.my_conf['gobal_epoch']):
        sr.train()
    # 保存模型
    sr.saveModel('myfirstmodel')

