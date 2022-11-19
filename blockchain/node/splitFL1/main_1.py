from blockchain.node.splitFL1.SPclient import SPclient
from blockchain.node.splitFL1.SPserver import SPserver

if __name__ == '__main__':
    print('m=600, training...')
    s = SPserver()
    cln = SPclient()
    cln.train(s.train)
