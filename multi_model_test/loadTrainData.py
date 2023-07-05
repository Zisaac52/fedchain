# 加载数据集

from torchvision import datasets, transforms

load_path = '../data'


# 加载mnist数据集
def load2MnistLoader(is_train=True):
    # test_loader = torch.utils.data.DataLoader(mydatasets, batch_size=batch_size,shuffle=True)
    mydatasets = datasets.MNIST(load_path, train=is_train, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
    # test_datasets = datasets.MNIST(load_path, train=False,
    #                                transform=transforms.Compose([transforms.ToTensor(),
    #                                                             transforms.Normalize((0.1307,), (0.3081,))]))
    return mydatasets


# 加载cifar10 训练集
def load2Cifar10Loader(is_train=True):
    train_data = datasets.CIFAR10(load_path, train=is_train, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    return train_data


# # 加载cifar10 测试集
# def load2Cifar10TestLoader():
#     test_data = datasets.CIFAR10(load_path, train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ]))
#     return test_data

def load_fashion_mnist(is_train=True):
    return datasets.FashionMNIST(load_path, train=is_train, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]), download = True)


def load_cifar100(is_train=True):
    return datasets.cifar.CIFAR100(load_path, train=is_train, transform=None, download=True)

