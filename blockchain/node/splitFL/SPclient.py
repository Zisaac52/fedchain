import logging
import sys

import torch

from blockchain.node.splitFL.splitmodel import mnist_Net_client
from fl.loadTrainData import load2MnistLoader
logger = logging.getLogger()


class SPclient:
    def __init__(self):
        self.model_cln = mnist_Net_client()
        if torch.cuda.is_available():
            self.model_cln.cuda()
        self.optimizer = torch.optim.SGD(self.model_cln.parameters(), lr=0.01, momentum=0.0001)
        datasets, _ = load2MnistLoader()
        self.train_loader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=True)

    # 客户端先训练，完成后传入到服务端计算剩余的东西
    def train(self, server_train):
        ep = 15
        path = './data/ep/'
        torch.save(self.model_cln.state_dict(), '{}client-0.pth'.format(path))
        self.model_cln.train()
        flag = False
        epoch = 0
        for i in range(ep):
            epoch += 1
            for data in self.train_loader:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                # 优化器优化模型
                self.optimizer.zero_grad()
                # 开始训练
                fx = self.model_cln(imgs)
                cln_fx = fx.clone().detach().requires_grad_(True)
                dfx = server_train(cln_fx, targets, flag, epoch)
                if flag:
                    flag = False
                fx.backward(dfx)
                self.optimizer.step()
            flag = True
            torch.save(self.model_cln.state_dict(), '{}client-{}.pth'.format(path, epoch))
            # acc, loss = evalmodel()
            # print("第{}轮, 准确率:{}, 损失值:{}".format(i, acc, loss))
        logger.info('{} - 训练完成，共{}趟'.format(sys._getframe().f_code.co_name, ep))