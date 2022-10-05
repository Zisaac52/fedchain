import csv

import torch

import config
from fl.client import Client
from fl.loadTrainData import load2MnistLoader, load2Cifar10Loader
from fl.model import mnist_Net
from fl.modelEval import model_eval
from fl.server import Server


def myEval(mod):
    model0 = torch.load('data/model/{}/network_{}.pth'.format('base', 'base'))
    acc, loss = model_eval(model0)
    print('gobal_epoch,Accuracy,loss')
    print('{},{},{}'.format(-1, acc, loss))
    for i in range(30):
        my_model = torch.load('data/model/gobal/{}/network_{}_{}.pth'.format(mod, mod, i))
        acc, loss = model_eval(my_model)
        print('{},{},{}'.format(i, acc, loss))
    print('{}模型评估完成'.format(mod))


def load_trainsets():
    if config.my_conf['dataset'].lower() == 'mnist':
        datasets, _ = load2MnistLoader()
    elif config.my_conf['dataset'].lower() == 'cifar':
        datasets = load2Cifar10Loader()
    else:
        raise ValueError('config.my_conf.dataset配置错误，无法找到！')
    return torch.utils.data.DataLoader(datasets, batch_size=64)


if __name__ == '__main__':
    # 创建一个服务
    sr = Server()
    model, version = sr.get_model()
    # 加入工作节点
    for i in range(config.my_conf['client.amount']):
        # 注册客户端，向每个节点分发模型
        sr.addClient(Client(model=model, mod_version=version, lr=config.my_conf['learn_rate'], client_id=i),
                    '127.0.0.1:808{}'.format(i))
    # 开始迭代训练
    for i in range(config.my_conf['gobal_epoch']):
        sr.train(i)
        # 保存模型
        # sr.saveModel('data/model/gobal/{}/network_{}_{}.pth'.format(config.my_conf['test_mod'], config.my_conf['test_mod'], i))
    print('训练完毕')

    # 评估训练完成的模型
    # myEval('os')
    # myEval('os')
    # myEval('os')
    # myEval('ns')
    # myEval('ns')
    # myEval('ns')
