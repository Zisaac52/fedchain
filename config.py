import torch

my_conf = {
    # client
    # 客户端总数
    'client.amount': 10,
    # 到达该数量的用户提交模型后将进行聚合
    # 'client.k': 5,
    # 本地迭代次数
    'local_epoch': 4,
    # 是否开启本地模型评估
    'local_OpenEval': False,
    # 梯度选择器（adam，sgd）
    'optimizer': 'sgd',

    # server
    # 全局迭代次数
    'gobal_epoch': 20,
    # 是否开启模型评估
    'openEval': True,

    # test
    # 开启训练模式 我的新方案ns，基准方案os 【ns，os】
    'isTest': False,
    'test_mod': 'os',

    # 版本均衡实验
    'test_client_id': [],

    # =3 表示每隔两代将test_client_id里的客户端停用一次
    'test_in_nepoch': 4,
    # 权重调整参数 xi in (0,1]
    'test_param_xi': 0.05,

    # 数据集(cifar,mnist)
    'dataset': 'mnist',
    # 学习率
    'learn_rate': 0.01,
    # 设备选择 cpu ，gpu，torch.device('cpu')
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 数据集批次
    'BATCH_SIZE': 64,
}
