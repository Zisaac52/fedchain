import json


class Configure:
    _DATA = None

    def __init__(self):
        self._getConf()
        pass

    # 加载Jon配置文件
    def _getConf(self):
        try:
            with open('config.json') as f:
                self._DATA = json.load(f)
        except OSError:
            print("配置文件加载失败")
        pass

    # 使用属性名称加载配置项
    def getConfigByAttr(self,attr=''):
        if attr == '':
            raise Exception("空的属性名，请检查")
        return self._DATA[attr]