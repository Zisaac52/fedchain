import logging

import torch

import config
from blockchain.start import startNode
from fl.client import Client
from fl.loadTrainData import load2MnistLoader, load2Cifar10Loader
from fl.modelEval import model_eval
from fl.server import Server

logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log',encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)  # 设置日志的级别
# fh.setFormatter(formatter)#设置的日志的输出
ch.setFormatter(formatter)
# logger.addHandler(fh) #logger对象可以添加多个fh和ch对象
logger.addHandler(ch)


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
    # 启动该节点
    startNode(attr='SN',port='8080')
    pass
