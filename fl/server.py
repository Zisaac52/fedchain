import copy

import torch
from torchvision import models
import config
from fl.model import mnist_Net
from fl.modelEval import model_eval


# 服务器端重复进行客户端采样、参数分发、参数聚合三个步骤。其中参数聚合和参数分发都只针对基础层。
class Server:
    def __init__(self):
        self._gobal_model = None
        self._diffval = None

        self._clientList = []
        self._clientlab = []
        # 第一次运行，初始化并分发模型
        self._initModel()
        # 对当前训练计数，用于识别当前学习状态
        self._count = 0

    def _initModel(self):
        if self._gobal_model is None:
            if config.my_conf['dataset'] == "cifar10":
                self._gobal_model = models.resnet18()
            if config.my_conf['dataset'] == "mnist":
                # self.gobal_model = models.mnasnet1_0()
                self._gobal_model = mnist_Net()
            else:
                raise ValueError("config.my_conf.dataset配置有误，无法找到该项！")
            self._gobal_model.to(config.my_conf['device'])

    # 统一分发模型
    # 循环遍历发送初始模型到client
    def dispatch(self):
        for i, cln in enumerate(self._clientList):
            cln.setModelFromServer(self._gobal_model, i)
        pass

    # 对diff进行累加
    def accumulator(self, diff):
        if self._diffval is None:
            # 深拷贝，防止意外更改
            self._diffval = copy.deepcopy(diff)
        else:
            for name, value in self._diffval.items():
                add_per_layer = diff[name]
                add_per_layer = add_per_layer.to(config.my_conf['device'])
                value = value.float()
                add_per_layer = add_per_layer.float()
                value.add_(add_per_layer)
        pass

    # 进行模型聚合，接收来自worknode的diff，更新至自己的模型
    def aggregation(self):
        # 全局模型参数更新,得到新的global_model
        for name, value in self._gobal_model.state_dict().items():
            update_per_layer = self._diffval[name]
            update_per_layer = update_per_layer.to(config.my_conf['device'])
            # 计算平均diff，将全部客户端的结果进行聚合
            update_per_layer = update_per_layer * (1 / len(self._clientList))
            value = value.float()
            update_per_layer = update_per_layer.float()
            value.add_(update_per_layer)
        # 聚合完成后将全局diff清空
        self._diffval = None

    # 保存每次聚合完成的模型
    def saveModel(self, filename):
        torch.save(self._gobal_model, "data/model/gobal/network_{}.pth".format(filename))

    def addClient(self, client, cln_name):
        self._clientList.append(client)
        self._clientlab.append(cln_name)
        print('节点-{}-加入'.format(cln_name))

    def train(self):
        # 分发初始模型
        self.dispatch()
        if self._count == 0:
            # 输出准确率和loss
            if config.my_conf['openEval']:
                print('gobal_epoch,Accuracy,loss')
        # 模型评估
        if config.my_conf['openEval']:
            acc, loss = model_eval(self._gobal_model)
            print('{},{},{}'.format(self._count, acc, loss))
        self._count += 1
        # 训练聚合
        for i, cln in enumerate(self._clientList):
            diff = cln.local_train()
            self.accumulator(diff)
        self.aggregation()
