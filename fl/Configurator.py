import torch
from munch import DefaultMunch


class Configurator:
    def __init__(self):
        self._config = DefaultMunch.fromDict(my_conf)

    def get_config(self):
        return self._config


my_conf = {
    # client
    # 客户端总数
    'client_amount': 10,
    # 随机选择客户端的数量
    'client_k': 4,
    # 本地迭代次数
    'local_epoch': 4,
    # 是否开启本地模型评估
    'local_OpenEval': False,
    # 梯度选择器（adam，sgd）
    'optimizer': 'sgd',
    # cifar 搞 0.01试试，mnist 0.0001
    'momentum': 0.95,

    # server
    # 全局迭代次数
    'gobal_epoch': 101,
    # 是否开启模型评估
    'openEval': True,

    # 是否开启异步聚合和调度的测试实验
    # 两个参数需要同步配置，值为True时test_client_id不为空
    'issyntest': True,
    'test_client_id': [4],

    # 数据集(cifar,mnist,fmnist,cifar100)
    'dataset': 'cifar100',
    # 是否加载现有模型进行训练
    'load_model': False,
    'load_path': './data/model/gobal/network_{}_{}_{}_{}_{}_{}.pth',
    # 学习率
    'learn_rate': 0.01,
    # 设备选择 cpu ，gpu，torch.device('cpu')
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 数据集批次
    'BATCH_SIZE': 64,
}
