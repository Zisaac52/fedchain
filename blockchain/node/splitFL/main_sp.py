from blockchain.node.splitFL.SPclient import SPclient
from blockchain.node.splitFL.SPeval import evalmodel
from blockchain.node.splitFL.SPserver import SPserver

if __name__ == '__main__':
    s = SPserver()
    cln = SPclient()
    cln.train(s.train)
    acc, loss = evalmodel(s.model_sev, cln.model_cln)
    print("准确率:{}, 损失值:{}".format(acc, loss))