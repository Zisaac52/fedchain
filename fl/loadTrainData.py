# 加载数据集

from torchvision import datasets, transforms

load_path = '../data'


def load2MnistLoader(is_train=True):
    # test_loader = torch.utils.data.DataLoader(mydatasets, batch_size=batch_size,shuffle=True)
    mydatasets = datasets.MNIST(load_path, train=is_train, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
    # test_datasets = datasets.MNIST(load_path, train=False,
    #                                transform=transforms.Compose([transforms.ToTensor(),
    #                                                             transforms.Normalize((0.1307,), (0.3081,))]))
    return mydatasets


# 加载cifar10 数据集
def load2Cifar10Loader(is_train=True):
    train_data = datasets.CIFAR10(load_path, train=is_train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    return train_data


def load_fashion_mnist(is_train=True):
    return datasets.FashionMNIST(load_path, train=is_train, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]), download = True)
