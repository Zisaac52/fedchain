import torch
from torchvision import models

import config
from blockchain.node.fed.model import mnist_Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 服务器端重复进行客户端采样、参数分发、参数聚合三个步骤
class Server:
    def __init__(self):
        self._gobal_model = None
        self._diffval = None
        self.version = 10
        # 第一次运行，初始化并分发模型
        self._initModel()

    def _initModel(self):
        if self._gobal_model is None:
            self._gobal_model = mnist_Net()
            self._gobal_model.to(device)

    # 进行模型聚合，接收来自worknode的diff，更新至自己的模型
    def aggregation(self):
        # 全局模型参数更新,得到新的global_model
        for name, value in self._gobal_model.state_dict().items():
            update_per_layer = self._diffval[name]
            update_per_layer = update_per_layer.to(device)
            # 计算平均diff，将全部客户端的结果进行聚合
            value = value.float()
            update_per_layer = update_per_layer.float()
            value.add_(update_per_layer)
        # 聚合完成后将全局diff清空
        self._diffval = None

    # 保存每次聚合完成的模型
    def saveModel(self, path):
        torch.save(self._gobal_model, path)
        print('保存模型，路径：{}'.format(path))

    # 原始的训练聚合方法
    def start_train(self, diff):
        if diff is not None:
            self._diffval = diff
            # 模型聚合
            self.aggregation()
            self.version += 1

    # 获取初始模型
    def get_model(self):
        return self._gobal_model, self.version

