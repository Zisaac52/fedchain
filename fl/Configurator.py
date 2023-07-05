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
    'client_amount': 5,
    # 本地迭代次数
    'local_epoch': 4,
    # 是否开启本地模型评估
    'local_OpenEval': False,
    # 梯度选择器（adam，sgd）
    'optimizer': 'sgd',

    # server
    # 全局迭代次数
    'gobal_epoch': 10,
    # 是否开启模型评估
    'openEval': True,
    'test_client_id': [],

    # 数据集(cifar,mnist)
    'dataset': 'mnist',
    # 学习率
    'learn_rate': 0.01,
    # 设备选择 cpu ，gpu，torch.device('cpu')
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 数据集批次
    'BATCH_SIZE': 64,
}
