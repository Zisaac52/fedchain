# 加载数据集

from torchvision import datasets, transforms


# 加载mnist数据集
def load2MnistLoader():
    # test_loader = torch.utils.data.DataLoader(mydatasets, batch_size=batch_size,shuffle=True)
    train_datasets = datasets.MNIST('data', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
    test_datasets = datasets.MNIST('data', train=False,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
    return train_datasets, test_datasets


# 加载cifar10 训练集
def load2Cifar10Loader():
    train_data = datasets.CIFAR10('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    return train_data


# 加载cifar10 测试集
def load2Cifar10TestLoader():
    test_data = datasets.CIFAR10('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return test_data
