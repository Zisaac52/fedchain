import logging

import torch

from blockchain.node.splitFL.splitmodel import mnist_Net_client
from fl.loadTrainData import load2MnistLoader

logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SPclient:
    def __init__(self):
        self.model_cln = mnist_Net_client()
        self.path = './data/ep1/'
        self.model_cln.load_state_dict(torch.load('{}client-0.pth'.format(self.path)))
        self.model_cln.to(device)
        self.optimizer = torch.optim.SGD(self.model_cln.parameters(), lr=0.001, momentum=0.0001)
        datasets, _ = load2MnistLoader()
        self.train_loader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=True)
        logger.info('Client initalization')

    # 客户端先训练，完成后传入到服务端计算剩余的东西
    def train(self, server_train):
        ep = 20
        self.model_cln.train()
        flag = False
        epoch = 0
        count = 0
        for i in range(ep):
            epoch += 1
            for data in self.train_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                count += len(imgs)
                # 优化器优化模型
                self.optimizer.zero_grad()
                # 开始训练
                fx = self.model_cln(imgs)
                cln_fx = fx.clone().detach().requires_grad_(True)
                dfx = server_train(cln_fx, targets, flag, count)
                if flag:
                    flag = False
                if dfx != 'Nodata':
                    # print('反向传播-{}'.format(count))
                    fx.backward(dfx)
                self.optimizer.step()
            flag = True
            torch.save(self.model_cln.state_dict(), '{}client-{}.pth'.format(self.path, epoch))
            # acc, loss = evalmodel()
            # print("第{}轮, 准确率:{}, 损失值:{}".format(i, acc, loss))
            print('第{}趟'.format(epoch))
        # logger.info('{} - 训练完成，共{}趟'.format(sys._getframe().f_code.co_name, ep))