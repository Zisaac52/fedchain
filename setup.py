import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms

from DLGAttack.attack import deep_leakage_from_gradients, showImg
from DLGAttack.dlg.models.vision import LeNet, weights_init
from DLGAttack.dlg.utils import label_to_onehot, cross_entropy_for_onehot

if __name__ == '__main__':
    print(torch.__version__, torchvision.__version__)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Running on %s" % device)

    dst = datasets.CIFAR10("data", download=True)
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    img_index = 30
    gt_data = tp(dst[img_index][0]).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label).to(device)

    plt.imshow(tt(gt_data[0].cpu()))

    net = LeNet().to(device)

    torch.manual_seed(1234)

    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # compute original gradient
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    h = deep_leakage_from_gradients(net, original_dy_dx, gt_data, gt_onehot_label)
    showImg(h)
