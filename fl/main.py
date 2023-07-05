import logging

from fl.Configurator import Configurator
from fl.client import Client
from fl.server import Server
logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log',encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)  # 设置日志的级别
# fh.setFormatter(formatter)#设置的日志的输出
ch.setFormatter(formatter)
# logger.addHandler(fh) #logger对象可以添加多个fh和ch对象
logger.addHandler(ch)

if __name__ == '__main__':
    config = Configurator().get_config()
    # 创建一个服务 
    sr = Server(config)
    model, version = sr.get_model()
    # 加入工作节点
    for i in range(config.client_amount):
        # 注册客户端，向每个节点分发模型
        sr.addClient(Client(conf=config, model=model, mod_version=version, client_id=i),
                    '127.0.0.1:808{}'.format(i))
    # 开始全局迭代训练
    for i in range(config.gobal_epoch):
        sr.start_train()
        # 保存模型
        # sr.saveModel('data/model/gobal/{}/network_{}_{}.pth'.format(config.my_conf['test_mod'], config.my_conf['test_mod'], i))
