import logging
import sys
import time

import torch

from fl.loadTrainData import load2MnistLoader
from fl.model import mnist_Net
logger = logging.getLogger()


def get_endNode_perfomance():
    logger.info('{} - Start evaluating performance...'.format(sys._getframe().f_code.co_name))
    model = mnist_Net()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0001)
    datasets, _ = load2MnistLoader()
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=True)
    # 模型训练逻辑
    model.train()
    # 加载数据集进行训练
    count = 1
    start = time.time()
    for data in train_loader:
        imgs, targets = data
        count += len(data)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 优化器优化模型
        optimizer.zero_grad()
        # 开始训练
        output = model(imgs)
        loss = torch.nn.functional.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()
        if count >= 1280:
            break
    end = time.time()
    logger.info('{} - 评估用时{}s'.format(sys._getframe().f_code.co_name, (end - start)))
    return count / (end - start)
