import copy
import time

import torch


class Client:
    """初始化：向服务器请求模型，更新为本地模型\n
    训练：记载数据集，开始训练\n
    每训练完一次向服务器上传diff参数"""
    def __init__(self, conf, model, mod_version, client_id=-1, datasets=None):
        self.config = conf
        self.optimizer = None
        self._client_id = client_id
        self.BATCH_SIZE = conf.BATCH_SIZE
        if self.BATCH_SIZE < 1:
            raise ValueError('BATCH_SIZE配置有误！')
        lr = getattr(conf, 'learn_rate', None)
        lr = lr if lr is not None else getattr(conf, 'learning_rate', None)
        self.learning_rate = 0.001 if lr is None else lr
        self.epoch = conf.local_epoch
        if self.epoch < 1:
            raise ValueError('local_epoch配置有误！')
        self.id = client_id
        self.device = conf.device
        # 加载数据集

        self._local_model = copy.deepcopy(model)
        self._gobal_model = copy.deepcopy(model)
        self.g_version = -1 if self.id in conf.test_client_id else mod_version

        self._dataLoader = self._randomLoad(datasets)
        self._dataset_size = getattr(self, '_dataset_size', 0)
        self._override_epoch = None
        self._override_batches = None
        self._last_elapsed = None

    # def load_trainsets(self):
    #     if self.config.dataset.lower() == 'mnist':
    #         datasets, _ = load2MnistLoader()
    #     elif self.config.dataset.lower() == 'cifar':
    #         datasets = load2Cifar10Loader()
    #     elif self.config.dataset.lower() == "fmnist":
    #         datasets = load_fashion_mnist(is_train=True)
    #     else:
    #         raise ValueError('config.my_conf.dataset配置错误，无法找到！')
    #     return datasets

    # 开始本地训练
    def local_train(self):
        start_time = time.time()
        self.optimizer = self._get_optimizer()
        # # 模型评估
        # if config.my_conf['local_OpenEval']:
        #     acc, loss = model_eval_nograde(self._local_model, self._dataLoader)
        #     print('{},{},{}'.format(self.id, acc, loss))
        # # 异步不平衡训练实验
        # if self.id in config.my_conf['test_client_id']:
        #     return None, -1

        # 模型训练逻辑
        self._local_model.train()
        local_epochs = max(1, int(self._override_epoch)) if self._override_epoch else self.epoch
        max_batches = None
        if self._override_batches is not None:
            max_batches = max(1, int(self._override_batches))

        for _ in range(local_epochs):
            batch_counter = 0
            for data in self._dataLoader:
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
                batch_counter += 1
                if max_batches is not None and batch_counter >= max_batches:
                    break

        self._override_epoch = None
        self._override_batches = None
        diff, version = self.getDiff()
        elapsed = time.time() - start_time
        self._last_elapsed = elapsed
        return diff, version, elapsed

    # 返回一个梯度选择器
    def _get_optimizer(self):
        if self.config.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self._local_model.parameters(), lr=self.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self._local_model.parameters(), lr=self.learning_rate, momentum=self.config.momentum)
        else:
            raise Exception('optimizer配置有误')
        return optimizer

    # 根据客户端数量平均分配数据集，然后随机打乱训练集，并返回加载器
    def _randomLoad(self, mydatasets):
        a_range = list(range(len(mydatasets)))
        datalen = int(len(mydatasets) / self.config.client_amount)
        if self.id != -1:
            trange = a_range[self.id * datalen:(self.id + 1) * datalen]
        else:
            raise Exception("client id error!")
        self._dataset_size = len(trange)
        # 构造数据器
        train_loader = torch.utils.data.DataLoader(mydatasets, batch_size=self.BATCH_SIZE,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(trange))
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

    def getDataset(self):
        return self._dataLoader

    def get_client_id(self):
        return self._client_id

    # --------------------------DDMLTS helpers
    def apply_workload(self, assignment):
        if assignment is None:
            self._override_epoch = None
            self._override_batches = None
            return
        self._override_epoch = assignment.get('local_epoch', self.epoch)
        self._override_batches = assignment.get('micro_batches')

    def get_dataset_size(self):
        return getattr(self, '_dataset_size', 0)

    def estimate_speed(self):
        elapsed = self._last_elapsed if self._last_elapsed is not None else 1.0
        dataset_size = self.get_dataset_size() or 1.0
        return dataset_size / max(elapsed, 1e-6)

    def get_state_vector(self):
        dataset_size = float(self.get_dataset_size() or 1.0)
        elapsed = float(self._last_elapsed or 1.0)
        speed = float(self.estimate_speed())
        return (dataset_size, elapsed, speed, speed)
