import torch

try:
    from Configurator import Configurator
    from loadTrainData import load2MnistLoader, load_fashion_mnist, load2Cifar10Loader, load2Cifar100Loader
except ImportError:
    from fl.Configurator import Configurator
    from fl.loadTrainData import load2MnistLoader, load_fashion_mnist, load2Cifar10Loader, load2Cifar100Loader

config = Configurator().get_config()
device = config.device


def get_test_loader():
    if config.dataset.lower() == "mnist":
        mydatasets = load2MnistLoader(is_train=False)
    elif config.dataset.lower() == "fmnist":
        mydatasets = load_fashion_mnist(is_train=False)
    elif config.dataset.lower() == "cifar":
        mydatasets = load2Cifar10Loader(is_train=False)
    elif config.dataset.lower() == "cifar100":
        mydatasets = load2Cifar100Loader(is_train=False)
    else:
        raise Exception("only support mnist，cifar and fmnist!")
    return torch.utils.data.DataLoader(mydatasets, batch_size=config.BATCH_SIZE, shuffle=True)


test_loaders = get_test_loader()


def _compute_classification_metrics(conf_matrix):
    eps = 1e-12
    tp = conf_matrix.diag().float()
    fp = conf_matrix.sum(dim=0).float() - tp
    fn = conf_matrix.sum(dim=1).float() - tp

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    precision = precision_per_class.mean().item() * 100.0
    recall = recall_per_class.mean().item() * 100.0
    f1 = f1_per_class.mean().item() * 100.0
    return precision, recall, f1


def model_eval(model, device, testLoader=None):
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
        testLoader = test_loaders
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    conf_matrix = None
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
        if conf_matrix is None:
            num_classes = output.shape[1]
            conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(target.view(-1).cpu(), pred.view(-1).cpu()):
            conf_matrix[t.long(), p.long()] += 1
    # torch.view_as(tensor)即将调用函数的变量，转变为同参数tensor同样的形状
    # torch.eq()对两个张量tensor进行逐元素比较，如果相等则返回True，否则返回False。True和False作运算时可以作1、0使用
    # .cpu()这一步将预测结果放到cpu上，利用电脑内存存储列表值。从而避免测试过程中爆显存。
    # .sum()是将我们一个批量的预测值求和，便于累加到correct变量中。
    # .item()取出 单元素张量的元素值 并返回该值，保持原元素类型不变。

    acc = 100.0 * (float(correct) / float(dataset_size))  # 准确率
    aver_loss = total_loss / dataset_size  # 平均损失
    precision = recall = f1 = 0.0
    if conf_matrix is not None:
        precision, recall, f1 = _compute_classification_metrics(conf_matrix)

    return acc, aver_loss, precision, recall, f1


# 返回一个元组类型（正确的个数，总的测试集数量，准确率，损失值）
def model_eval_nograde(model, testset=None):
    testLoader = test_loaders if testset is None else testset
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
    return (100. * correct / len(testLoader.dataset)), (test_loss / len(testLoader.dataset))
    # return (
    #     correct,
    #     len(testLoader.dataset),
    #     100. * correct / len(testLoader.dataset),
    #     test_loss / len(testLoader.dataset)
    # )
