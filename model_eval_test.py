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
        stictsp = torch.load('E:/研究生/研究/AIOT/实验/mnist的SPFL与FL的卸载效果对比/0.001-20-32/sp/server-{}.pth'.format(i))
        stictep = torch.load('E:/研究生/研究/AIOT/实验/mnist的SPFL与FL的卸载效果对比/0.001-20-32/ep/client-{}.pth'.format(i))
        model_sev.load_state_dict(stictsp)
        model_cln.load_state_dict(stictep)
        # if torch.cuda.is_available():
        #     model_sev.cuda()
        #     model_cln.cuda()
        acc1, loss1 = evalmodel(model_sev, model_cln)
        print('{},{}'.format(acc1, loss1))


if __name__ == '__main__':
    print("start...")
    print("acc  loss")
    # print("10.11	0.072213045\n94.99	0.005088198\n95.28	0.005083745")
    # print("96.93	0.003155656\n97.52	0.002512243\n97.99	0.002083545\n98.21	0.001910615\n98.03	0.002022026")
    # print("98.35	0.001691023\n98.4	0.00167461\n98.54	0.001537103\n98.44	0.001593247\n98.67	0.001382278")
    # print("98.49	0.001427062\n98.61	0.001314504\n98.58	0.001408369")
    # testSpfl()
    testFL()
