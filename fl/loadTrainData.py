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
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_data = datasets.CIFAR10(load_path, train=is_train, download=True,
                                transform=transform)
    return train_data


def load_fashion_mnist(is_train=True):
    return datasets.FashionMNIST(load_path, train=is_train, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]), download = True)
