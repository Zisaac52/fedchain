import copy
import torch
import config
from fl.loadTrainData import load2MnistLoader, load2Cifar10Loader
from fl.modelEval import model_eval, model_eval_nograde


# 初始化：向服务器请求模型，更新为本地模型
# 训练：记载数据集，开始训练
# 每训练完一次向服务器上传diff参数
def load_trainsets():
    if config.my_conf['dataset'].lower() == 'mnist':
        datasets, _ = load2MnistLoader()
    elif config.my_conf['dataset'].lower() == 'cifar':
        datasets = load2Cifar10Loader()
    else:
        raise ValueError('config.my_conf.dataset配置错误，无法找到！')
    return datasets


class Client:
    def __init__(self, lr=0.001):
        self.dataLoader = None
        self.optimizer = None
        self.BATCH_SIZE = config.my_conf['BATCH_SIZE']
        if self.BATCH_SIZE < 1:
            raise ValueError('BATCH_SIZE配置有误！')
        self.learning_rate = lr
        self.epoch = config.my_conf['local_epoch']
        if self.epoch < 1:
            raise ValueError('local_epoch配置有误！')
        self.id = -1
        self.datasets = load_trainsets()
        self.device = config.my_conf['device']
        # 加载数据集

        self._local_model = None
        self._gobal_model = None

    # 开始本地训练
    def local_train(self):
        self.optimizer = self._get_optimizer()
        dataLoader = self._randomLoad(self.datasets)
        self._local_model.train()
        for i in range(self.epoch):
            # 加载数据集进行训练
            for data in dataLoader:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                # 优化器优化模型
                self.optimizer.zero_grad()
                # 开始训练
                output = self._local_model(imgs)
                loss = torch.nn.functional.cross_entropy(output, targets)
                loss.backward()
                self.optimizer.step()
            # 模型评估
            if config.my_conf['local_OpenEval']:
                acc, loss = model_eval_nograde(self._local_model)
                print('{},{},{}'.format(self.id, acc, loss))
        diff = self.getDiff()
        return diff

    # 返回一个梯度选择器
    def _get_optimizer(self):
        if config.my_conf['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(self._local_model.parameters(), lr=self.learning_rate)
        elif config.my_conf['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(self._local_model.parameters(), lr=self.learning_rate, momentum=0.0001)
        else:
            raise Exception('optimizer配置有误')
        return optimizer

    # 根据客户端数量平均分配数据集，然后随机打乱训练集，并返回加载器
    def _randomLoad(self, mydatasets):
        a_range = list(range(len(mydatasets)))
        datalen = int(len(mydatasets) / config.my_conf['client.amount'])
        if self.id != -1:
            trange = a_range[self.id * datalen:(self.id + 1) * datalen]
        else:
            raise Exception("client id error!")
        # 构造数据器
        train_loader = torch.utils.data.DataLoader(mydatasets, batch_size=self.BATCH_SIZE,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(trange))
        return train_loader

    def getDiff(self):
        diff = dict()
        # 遍历更新之后的各层模型参数。并返回每层对应的名字(name)和数据。
        for name, data in self._local_model.state_dict().items():
            # print(data != model.state_dict()[name])  # 用于打印出来是否参数相等
            # 将当前name和全局模型所对应name的数据进行相减，得到权重大小的变化量即权重差
            diff[name] = (data - self._gobal_model.state_dict()[name])
        return diff

    # 接收来自服务端的模型,由服务器调用设置模型
    def setModelFromServer(self, model, ids):
        self._gobal_model = copy.deepcopy(model)
        self._local_model = copy.deepcopy(model)
        if config.my_conf['local_OpenEval']:
            print('gobal_epoch,Accuracy,loss')
            acc, loss = model_eval_nograde(self._local_model)
            print('{},{},{}'.format(ids, acc, loss))
        self.id = ids
        # if torch.cuda.is_available():
        #     self._local_model.to(self.device)
        #     self._gobal_model.to(self.device)
