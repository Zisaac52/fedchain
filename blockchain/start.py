import logging

from blockchain.node.node_en import NodeEN
from blockchain.node.node_sn import NodeSN

logger = logging.getLogger()


# 判断启动节点的类型，按照相应的启动方式运行
def startNode(port='', attr=''):
    if attr.upper() == "EN":
        node = NodeEN(port=port)
    elif attr.upper() == "SN":
        node = NodeSN(port=port)
    else:
        raise ValueError("The input parameters are incorrect, there are only two param->(SN,EN)")
    try:
        node.startNode()
    except RuntimeError as e:
        logger.error(e)

