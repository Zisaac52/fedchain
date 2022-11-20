import logging

import torch

from blockchain.node.splitFL.splitmodel import mnist_Net_server
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ldevice = 'cuda' if torch.cuda.is_available() else 'cpu'


class SPserver:
    model_sev = None

    def __init__(self):
        self.model_sev = mnist_Net_server()
        self.path = './data/sp1/'
        self.model_sev.load_state_dict(torch.load('{}server-0.pth'.format(self.path), map_location=ldevice))
        self.model_sev.to(device)
        self.optimizer = torch.optim.SGD(self.model_sev.parameters(), lr=0.01, momentum=0.0001)
        self.epoch = 0
        logger.info('Client initalization')

    def train(self, fx_client, cln_targets, flag, count):
        self.model_sev.train()
        self.optimizer.zero_grad()
        output = self.model_sev(fx_client)
        loss = torch.nn.functional.cross_entropy(output, cln_targets)
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer.step()
        if flag:
            self.epoch += 1
            torch.save(self.model_sev.state_dict(), '{}server-{}.pth'.format(self.path, self.epoch))
        if count > 0 and count % 10 == 0:
            return dfx_client
        else:
            return 'Nodata'
