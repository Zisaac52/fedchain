import torch

from blockchain.node.splitFL.splitmodel import mnist_Net_server


class SPserver:
    model_sev = None

    def __init__(self):
        self.model_sev = mnist_Net_server()
        if torch.cuda.is_available():
            self.model_sev.cuda()
        self.optimizer = torch.optim.SGD(self.model_sev.parameters(), lr=0.01, momentum=0.0001)
        self.count = 0

    def train(self, fx_client, cln_targets, isSave=False):
        self.model_sev.train()
        self.optimizer.zero_grad()
        output = self.model_sev(fx_client)
        loss = torch.nn.functional.cross_entropy(output, cln_targets)
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer.step()
        self.count += 32
        # if self.count % 320 == 0:
        #     torch.save(self.model_sev.state_dict(), './data/sp/server-{}.pth'.format(self.count))
        if isSave:
            return dfx_client, self.model_sev
        else:
            return dfx_client, None
