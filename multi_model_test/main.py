import logging
import os
import sys
import time

import torch
import torchvision.models

from loadTrainData import load2MnistLoader, load2Cifar10Loader, load_fashion_mnist, load_cifar100
from models import mnist_Net


logger = logging.getLogger()
# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler('test1.log', encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -m %(message)s')
# 设置日志的级别
logger.setLevel(logging.INFO)
# 设置的日志的输出
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# logger对象可以添加多个fh和ch对象
# logger.addHandler(fh)
logger.addHandler(ch)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def load_trainsets(config):
    if config.lower() == 'mnist':
        datasets = load2MnistLoader()
    elif config.lower() == 'cifar':
        datasets = load2Cifar10Loader()
    else:
        raise ValueError('config.my_conf.dataset配置错误，无法找到！')
    return datasets


def load_dataset(mydataset, is_training=True):
    dictconfig = {
        'mnist': load2MnistLoader,
        'cifar10': load2Cifar10Loader,
        'fashion': load_fashion_mnist,
        'cifar100': load_cifar100
    }

    datasets = dictconfig.get(mydataset.lower(), None)
    if datasets is not None:
        return datasets(is_training)
    else:
        raise ValueError(f' 未找到数据集{mydataset}，请检查参数配置！')


def load_model(dataset):
    '''
    :param dataset: mnist | fashion | cifar10 | cifar100
    :return: 对应的训练模型
    '''
    dataset = dataset.lower()
    if 'mnist' == dataset or 'fashion' == dataset:
        xxmodel = mnist_Net()
    elif 'cifar10' == dataset:
        xxmodel = torchvision.models.resnet18()
    elif 'cifar100' == dataset:
        xxmodel = torchvision.models.resnet34()
    else:
        raise ValueError("No such model!")
    return xxmodel


def model_eval(model, device, testLoader=None ):
    """用于评估模型准确率和损失值的\n
    传入模型和测试集\n
    :param device: cuda | cpu
    :param model 待评估模型\n
    :param testLoader 测试集加载器\n
    :returns aver_loss,accuracy\n
    # 进入模型评估模式"""
    model.eval()
    # 如果没传参，就用原来的全集
    if testLoader is None:
        raise ValueError("Testset is None")
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for batch_id, batch in enumerate(testLoader):  # batch_id就为enumerate()遍历集合所返回的批量序号
        inputs, target = batch  # 得到数据集和标签
        inputs = inputs.to(device)
        target = target.to(device)
        dataset_size += inputs.size()[0]  # data.size()=[batch,通道数,32,32]、target.size()=[batch]
        output = model(inputs)

        total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        # if config.my_conf["dataset"].lower() == "mnist":
        #     total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        # elif config.my_conf["dataset"].lower() == "cifar":
        #     total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        # else:
        #     raise TypeError("Not find Appropriate mode.")
        # .data意即将变量的tensor取出来
        # 因为tensor包含data和grad，分别放置数据和计算的梯度
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        # 按照从左往右的 第一维 取出最大值的索引 torch.max()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    # torch.view_as(tensor)即将调用函数的变量，转变为同参数tensor同样的形状
    # torch.eq()对两个张量tensor进行逐元素比较，如果相等则返回True，否则返回False。True和False作运算时可以作1、0使用
    # .cpu()这一步将预测结果放到cpu上，利用电脑内存存储列表值。从而避免测试过程中爆显存。
    # .sum()是将我们一个批量的预测值求和，便于累加到correct变量中。
    # .item()取出 单元素张量的元素值 并返回该值，保持原元素类型不变。

    acc = 100.0 * (float(correct) / float(dataset_size))  # 准确率
    aver_loss = total_loss / dataset_size  # 平均损失

    return acc, aver_loss


def train(Dts):
    learning_rate = Dts.lr
    epoch = Dts.epoch
    dataset_name = Dts.name
    BATCH_SIZE = Dts.batch_size
    logger.info(f'dataset_name:{dataset_name},learning_rate:{learning_rate},epoch:{epoch},BATCH_SIZE:{BATCH_SIZE}')
    train_loader = torch.utils.data.DataLoader(load_dataset(dataset_name), batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=0)
    test_loader = torch.utils.data.DataLoader(load_dataset(dataset_name, False), batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=0)
    device = torch.device('cpu')
    model = load_model(dataset_name)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0001)
    logger.info('ep,acc,loss,run time(s),speed(s/p)')
    # 模型训练逻辑
    model.train()
    for i in range(epoch):
        # 加载数据集进行训练
        count = 0
        start = time.time()
        for data in train_loader:
            imgs, targets = data
            count += len(data)
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 优化器优化模型
            optimizer.zero_grad()
            # 开始训练
            output = model(imgs)
            loss = torch.nn.functional.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
        # 统计数据跑完一轮的时间
        end = time.time()
        acc, loss = model_eval(model, device, testLoader=test_loader)
        logger.info(f'{i},{acc},{loss},{end - start},{(end - start) / count}')


class DtsConig:
    def __init__(self, name, lr, batch_size, epoch):
        self.name = name
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch


if __name__ == '__main__':
    logger.info('训练数据集：mnist | fashion | cifar10 | cifar100')
    arr = [DtsConig('mnist', 0.001, 64, 80),
            DtsConig('fashion', 0.001, 64, 80),
            DtsConig('cifar10', 0.001, 64, 100),
            DtsConig('cifar100', 0.001, 64, 100)]
    for Dts in arr:
        train(Dts)

