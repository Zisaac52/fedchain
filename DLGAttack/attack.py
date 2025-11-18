import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms

import config
from DLGAttack.dlg.models.vision import LeNet, weights_init
from fl.loadTrainData import load2MnistLoader, load2Cifar10Loader


def load_trainsets():
    if config.my_conf['dataset'].lower() == 'mnist':
        datasets = load2MnistLoader()
    elif config.my_conf['dataset'].lower() == 'cifar':
        datasets = load2Cifar10Loader()
    else:
        raise ValueError('config.my_conf.dataset配置错误，无法找到！')
    return torch.utils.data.DataLoader(datasets, batch_size=64)


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def deep_leakage_from_gradients(model, origin_grad, origin_data, dummy_label):
    if torch.cuda.is_available():
        model.cuda()
    tt = transforms.ToPILImage()
    criterion = cross_entropy_for_onehot
    dummy_data = torch.randn(origin_data.size()).cuda()
    dummy_label = torch.randn(dummy_label.size()).cuda()
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_grad, origin_grad):
                grad_diff += ((gx - gy) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            history.append(tt(dummy_data[0].cpu()))

    return history


def showImg(history):
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    plt.show()
