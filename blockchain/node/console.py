import logging
import sys

from blockchain.node.service.client import runRemoteFunc

logger = logging.getLogger()


def console_cmd(entry):
    """
    消息组成：{task: '', data: {}, tag1: '', tag2: ''....}\n
    任务类型-> 训练T，停止P，保存模型S，评估模型E \n
    :param entry:
    :return:
    """
    # 测试网络是否通畅，是否为可连接点
    try:
        msg = {'type': 10, 'status': 200, 'content': {'message': 'test'}}
        resp = runRemoteFunc('communicate', data=msg, HOST=entry.get('ip'),
                            PORT=entry.get('port'))
        if resp.get('status') != 200:
            raise ConnectionRefusedError('Remote node error！')
    except ConnectionError as e:
        print(e)
        return
    while True:
        cmd = input('[Console({}:{})]$'.format(entry.get('ip'), entry.get('port')))
        if cmd.lower() == 'quit()' or cmd.lower() == 'exit()':
            sys.exit(0)
        op = switch(cmd)
        if op is None:
            print("参数错误！")
        else:
            msg = {'type': 6, 'status': 200, 'content': {'task': op}}
            resp = runRemoteFunc('communicate', data=msg, HOST=entry.get('ip'),
                            PORT=entry.get('port'))
            print(resp)


def switch(opt):
    option = {
        # 训练模型
        'train()': 'T',
        # 暂停
        'pause()': 'P',
        # 保存
        'save()': 'S',
        # 评估模型
        'eval()': 'E',
        # 获取SN列表
        'getSN()': 'GSN',
        # 下载服务端模型
        'getModel()': 'DL',
        # 获取状态向量
        'getVector()': 'SV',
        # 查看DDMLTS调度计划
        'getSchedule()': 'SCH',
        # 开启联邦学习训练
        'trainFL()': 'FLT'
    }
    opts = option.get(opt)
    if opts != '':
        return opts
    else:
        return None
