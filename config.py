import torch

my_conf = {
    # client
    # 客户端总数
    'client.amount': 5,
    # 到达该数量的用户提交模型后将进行聚合
    # 'client.k': 5,
    # 本地迭代次数
    'local_epoch': 2,
    # 是否开启本地模型评估
    'local_OpenEval': False,
    # 梯度选择器（adam，sgd）
    'optimizer': 'sgd',

    # server
    # 全局迭代次数
    'gobal_epoch': 20,
    # 是否开启模型评估
    'openEval': True,

    # 数据集(cifar,mnist)
    'dataset': 'mnist',
    # 学习率
    'learn_rate': 0.01,
    # 设备选择 cpu ，gpu
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 数据集批次
    'BATCH_SIZE': 64,
}
