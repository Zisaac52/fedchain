import time


class Block:
    _block = None

    # 初始化，创建区块并赋值
    def __init__(self, mytime='', height=-1, mesagge='', preHash='', selfHash=''):
        currT = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        if height != -1 or preHash != '' or selfHash != '':
            self._block = dict()
            self._block['time'] = mytime
            self._block['createTime'] = currT
            self._block['height'] = height
            self._block['mesagge'] = mesagge
            self._block['preHash'] = preHash
            self._block['selfHash'] = selfHash
        else:
            raise Exception('The block cannot be empty!')

    def getBlock(self):
        return self._block
