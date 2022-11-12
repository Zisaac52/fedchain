import copy
import time

import torch
import config
from fl.loadTrainData import load2MnistLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 初始化：向服务器请求模型，更新为本地模型
# 训练：记载数据集，开始训练
# 每训练完一次向服务器上传diff参数
def load_trainsets():
    datasets, _ = load2MnistLoader()
    return datasets


class Client:
    def __init__(self, model, mod_version, lr=0.001):
        self.optimizer = None
        self.BATCH_SIZE = 32
        self.datasets = load_trainsets()
        # 加载数据集
        self._local_model = copy.deepcopy(model)
        self._gobal_model = copy.deepcopy(model)
        self.g_version = mod_version
        self.learning_rate = lr
        self.id = 0
        self._dataLoader = self._randomLoad(self.datasets)

    # 开始本地训练
    def local_train(self):
        self.optimizer = self._get_optimizer()
        # 模型训练逻辑
        self._local_model.train()
        for i in range(1):
            # 加载数据集进行训练
            for data in self._dataLoader:
                imgs, targets = data
                # count += len(data)
                if torch.cuda.is_available():
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                # 优化器优化模型
                self.optimizer.zero_grad()
                # 开始训练
                output = self._local_model(imgs)
                loss = torch.nn.functional.cross_entropy(output, targets)
                loss.backward()
                self.optimizer.step()
        diff = self.getDiff()
        return diff

    # 返回一个梯度选择器
    def _get_optimizer(self):
        optimizer = torch.optim.SGD(self._local_model.parameters(), lr=self.learning_rate, momentum=0.0001)
        return optimizer

    # 根据客户端数量平均分配数据集，然后随机打乱训练集，并返回加载器
    def _randomLoad(self, mydatasets):
        train_loader = torch.utils.data.DataLoader(mydatasets, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_loader

    # 将训练完成的模型发送到服务器聚合，带本地接收到的全局模型的版本
    def getDiff(self):
        diff = dict()
        # 遍历更新之后的各层模型参数。并返回每层对应的名字(name)和数据。
        for name, data in self._local_model.state_dict().items():
            # print(data != model.state_dict()[name])  # 用于打印出来是否参数相等
            # 将当前name和全局模型所对应name的数据进行相减，得到权重大小的变化量即权重差
            diff[name] = (data - self._gobal_model.state_dict()[name])

        return diff, self.g_version

    # 接收来自服务端的模型,由服务器调用设置模型
    def setModelFromServer(self, model, g_version):
        self._gobal_model = copy.deepcopy(model)
        self._local_model = copy.deepcopy(model)
        self.g_version = g_version

