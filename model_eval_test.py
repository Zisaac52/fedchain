import torch

from blockchain.node.splitFL.SPeval import evalmodel
from blockchain.node.splitFL.splitmodel import mnist_Net_server, mnist_Net_client
from fl.model import mnist_Net
from fl.modelEval import model_eval


def testFL():
    for i in range(20):
        model = mnist_Net()
        stict = torch.load('data/flm/gmodel-{}.pth'.format(i))
        model.load_state_dict(stict)
        if torch.cuda.is_available():
            model.cuda()
        acc, loss = model_eval(model)
        print('{},{}'.format(acc, loss))


def testSpfl():
    for i in range(20):
        model_sev = mnist_Net_server()
        model_cln = mnist_Net_client()
        stictsp = torch.load('E:/研究生/研究/AIOT/实验/mnist的SPFL与FL的卸载效果对比/0.001-20-32-1/sp/server-{}.pth'.format(i))
        stictep = torch.load('E:/研究生/研究/AIOT/实验/mnist的SPFL与FL的卸载效果对比/0.001-20-32-1/ep/client-{}.pth'.format(i))
        model_sev.load_state_dict(stictsp)
        model_cln.load_state_dict(stictep)
        if torch.cuda.is_available():
            model_sev.cuda()
            model_cln.cuda()
        acc1, loss1 = evalmodel(model_sev, model_cln)
        print('{},{}'.format(acc1, loss1))


if __name__ == '__main__':
    testSpfl()
    # testFL()
