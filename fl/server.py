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
        self.version = 10
        self._clientList = []
        self._clientlab = []
        # 第一次运行，初始化并分发模型
        self._initModel()
        # 对当前训练计数，用于识别当前学习状态
        self._count = 0

    def _initModel(self):
        if self._gobal_model is None:
            if config.my_conf['dataset'] == "cifar":
                self._gobal_model = models.resnet18()
            elif config.my_conf['dataset'] == "mnist":
                # self.gobal_model = models.mnasnet1_0()
                # self._gobal_model = mnist_Net()
                # 使用预训练的模型
                self._gobal_model = torch.load("data/model/base/network_base.pth")
            else:
                raise ValueError("config.my_conf.dataset配置有误，无法找到该项！")
            self._gobal_model.to(config.my_conf['device'])

    # 统一分发模型
    # 循环遍历发送初始模型到client
    def dispatch(self, cln):
        cln.setModelFromServer(self._gobal_model, self.version)

    # 对diff以权重u进行累加
    def accumulator(self, diff, u_w):
        if self._diffval is None:
            # 深拷贝，防止意外更改
            self._diffval = copy.deepcopy(diff)
        else:
            for name, value in self._diffval.items():
                add_per_layer = diff[name] * u_w
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
    def saveModel(self, path):
        torch.save(self._gobal_model, path)
        print('保存模型，路径：{}'.format(path))

    def addClient(self, client, cln_name):
        self._clientList.append(client)
        self._clientlab.append(cln_name)
        print('节点-{}-加入'.format(cln_name))

    # 原始的训练聚合方法
    def start_train(self):
        if self._count == 0:
            # 输出准确率和loss
            if config.my_conf['openEval']:
                print('gobal_epoch,Accuracy,loss')
        # 模型评估
        if config.my_conf['openEval']:
            acc, loss = model_eval(self._gobal_model)
            print('{},{},{}'.format(self._count, acc, loss))
        self._count += 1
        for i, cln in enumerate(self._clientList):
            diff, ver = cln.local_train()
            if ver != -1:
                self.accumulator(diff, 1)
        # 模型聚合
        self.aggregation()
        self.version += 1
        self.dispatch_normal()

    # 测试聚合方法
    def train(self, currentEpoch):

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
            skip_flag = False
            u = 1
            if config.my_conf['isTest']:
                skip_flag = self.test_my_scheme(i, currentEpoch)
                if skip_flag:
                    continue
                else:
                    diff, ver = cln.local_train()
                # 全局模型分发错误，大部分节点都落后一个版本
                # 若是新方案则进行更改u的值
                # print('第{}轮，客户端{}版本差异：{}'.format(currentEpoch, i, ver != self.version))
                if config.my_conf['test_mod'].lower() == 'ns':
                    if ver != self.version:
                        u = config.my_conf['test_param_xi'] / (self.version - ver)
                    self.accumulator(diff, u)
                    # print('全局第{}次迭代，当前全局模型版本{}，client-{}与全局版本差距{}'.format(currentEpoch, self.version, i, self.version - ver))
                if config.my_conf['test_mod'].lower() == 'os':
                    if not skip_flag:
                        self.accumulator(diff, u)
            else:
                diff, ver = cln.local_train()
                self.accumulator(diff, u)

        # 模型聚合
        self.aggregation()
        self.version += 1
        if config.my_conf['isTest']:
            # 在间隔0,3,6,9的时候分发模型
            self.dispatch_test(currentEpoch)
        else:
            self.dispatch_normal()

    # 获取初始模型
    def get_model(self):
        return self._gobal_model, self.version

    def test_my_scheme(self, i, currentEpoch):
        skip_flag = False
        # 训练聚合
        u = 1
        if currentEpoch % config.my_conf['test_in_nepoch'] != 0:
            # 跳过配置的客户端
            for j in config.my_conf['test_client_id']:
                # 在 test_in_nepoch 代训练 0, 3, 6, 9
                if i == j:
                    # 跳出循环，忽略该客户端本次更新
                    return True
        return skip_flag

    def dispatch_test(self, currentEpoch):
        # 对客户端进行筛选，其他正常客户端无需判断
        cln_id = [x for x in range(len(self._clientList))]
        cln_otr = list(set(cln_id) ^ set(config.my_conf['test_client_id']))
        for clnid in cln_otr:
            self.dispatch(self._clientList[clnid])
        if currentEpoch % config.my_conf['test_in_nepoch'] == 0:
            for ids in config.my_conf['test_client_id']:
                self.dispatch(self._clientList[ids])
        # 用聚合的模型进行评估
        for ids in config.my_conf['test_client_id']:
            acc, loss = model_eval(self._gobal_model, self._clientList[ids].getDataset())
            print('节点id{},{},{}'.format(ids, acc, loss))

    def dispatch_normal(self):
        for cln in self._clientList:
            self.dispatch(cln)
