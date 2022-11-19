import torch

from blockchain.node.splitFL.splitmodel import mnist_Net_server
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class SPserver:
    model_sev = None

    def __init__(self):
        self.model_sev = mnist_Net_server()
        self.path = './data/sp/'
        self.model_sev.load_state_dict(torch.load('{}server-0.pth'.format(self.path)))
        self.model_sev.to(device)
        self.optimizer = torch.optim.SGD(self.model_sev.parameters(), lr=0.01, momentum=0.0001)
        self.epoch = 0

    def train(self, fx_client, cln_targets, flag=False, epoch=0):
        self.model_sev.train()
        self.optimizer.zero_grad()
        output = self.model_sev(fx_client)
        loss = torch.nn.functional.cross_entropy(output, cln_targets)
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer.step()
        # if self.count % 320 == 0:
        #     torch.save(self.model_sev.state_dict(), './data/sp/server-{}.pth'.format(self.count))
        if flag:
            torch.save(self.model_sev.state_dict(), './data/sp/server-{}.pth'.format(self.epoch))
        return dfx_client
