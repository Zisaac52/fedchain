import torch
from torch import nn

import config
from fl.loadTrainData import load2MnistLoader, load2Cifar10Loader, load2Cifar10TestLoader
from fl.model import mnist_Net, ResNet18
from fl.modelEval import model_eval


def train():
    model = None
    datasets = None
    if config.my_conf['dataset'].lower() == 'mnist':
        model = mnist_Net()
        datasets = load2MnistLoader()
    elif config.my_conf['dataset'].lower() == 'cifar':
        model = ResNet18()
        datasets = load2Cifar10Loader()
    epoch = config.my_conf['gobal_epoch']

    # loss_fn = nn.functional.cross_entropy
    if torch.cuda.is_available():
        model.cuda()
    if config.my_conf['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.my_conf['learn_rate'])
    elif config.my_conf['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.my_conf['learn_rate'], momentum=0.0001)
    else:
        raise Exception('optimizer配置有误')

    if config.my_conf['local_OpenEval']:
        acc, loss, precision, recall, f1 = model_eval(model)
        print('gobal_epoch, Accuracy, loss, Precision, Recall, F1-score')
        print('{}, {}, {}, {}, {}, {}'.format(0, acc, loss, precision, recall, f1))
        # test(model, config.my_conf['device'])

    dataLoader = randomLoad(datasets, 10)
    model.train()

    for i in range(epoch):
        # print(f"------------第{i + 1}轮训练-------------")
        for data in dataLoader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            # 开始训练
            output = model(imgs)
            loss = nn.functional.cross_entropy(output, targets)
            # 优化器优化模型,梯度清零
            optimizer.zero_grad()
            loss.backward()
            # 利用反向传播得到的梯度，利用优化算法更新网络参数（权重）
            optimizer.step()

        if config.my_conf['local_OpenEval']:
            acc, loss, precision, recall, f1 = model_eval(model)
            print('{}, {}, {}, {}, {}, {}'.format(i + 1, acc, loss, precision, recall, f1))
            # test(model,config.my_conf['device'])


def myNewTrain():
    model = ResNet18()
    epoch = config.my_conf['gobal_epoch']
    datasets = load2Cifar10Loader()
    dataLoader = randomLoad(datasets, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.my_conf['learn_rate'], momentum=0.0001)
    # 本地模型训练
    model.train()
    for e in range(epoch):

        for batch_id, batch in enumerate(dataLoader):
            data, target = batch

            # for name, layer in self.local_model.named_parameters():
            #    print(torch.mean(self.local_model.state_dict()[name].data))
            # print("\n\n")

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            optimizer.step()


def test(model, device):
    model.eval()

    if config.my_conf["dataset"].lower() == "mnist":
        _, mydatasets = load2MnistLoader()
        testLoader = torch.utils.data.DataLoader(mydatasets, batch_size=config.my_conf['BATCH_SIZE'], shuffle=True)
    elif config.my_conf["dataset"].lower() == "cifar":
        mydatasets = load2Cifar10TestLoader()
        testLoader = torch.utils.data.DataLoader(mydatasets, batch_size=config.my_conf['BATCH_SIZE'], shuffle=True)
    else:
        raise Exception("only support mnist!")

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if config.my_conf["dataset"].lower() == "mnist":
                test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            elif config.my_conf["dataset"].lower() == "cifar":
                test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            else:
                raise TypeError("Not find Appropriate mode.")
    aver_loss = test_loss / len(testLoader.dataset)
    print('\nTest set:  Accuracy: {}/{} ({:.2f}%) avg_loss: {}'.format(
        correct,
        len(testLoader.dataset),
        100. * correct / len(testLoader.dataset), aver_loss)
    )


def randomLoad(mydatasets, n):
    id = 0
    a_range = list(range(len(mydatasets)))
    datalen = int(len(mydatasets) / n)  # config.my_conf['client.amount']
    trange = a_range[id * datalen:(id + 1) * datalen]

    # 构造数据器
    train_loader = torch.utils.data.DataLoader(mydatasets, batch_size=config.my_conf['BATCH_SIZE'], shuffle=False,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(trange))
    return train_loader
