# 联邦学习与区块链
环境：python 3.7
## 分布式网络仿真
使用grpc框架实现rpc通信，建立区块链节点
由于资源有限，利用mysql模拟区块存储

必要的组件库->[镜像源使用说明](https://blog.csdn.net/weixin_45523107/article/details/116535445)
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt
```

proto文件编译命令，在proto文件所在位置运行该命令
```
python -m grpc_tools.protoc -I. --python_out=./base_package --grpc_python_out=./base_package ./proto/data.proto
```
## 联邦学习
pytorch 1.12
本框架实现的是Personality federate learning联邦学习，总体流程如下：
1. 服务器端初始化基本层权重Wb。
2. 客户端初始化自己的个性化层权重Wp
3. 服务器端将Wb发送到各个客户端。
4. 服务器端和客户端都执行相同的循环：客户端收到服务器传来的基本层权重，然后利用本地数据来进行SGD更新：
5. 本地训练完毕后只将基本层权重传回服务器。
6. 服务器聚合所有客户端的基本层权重，然后分发给所有客户端，进行下一轮通信。

FedPer与FedAvg有以下3点不同：
1. FedAvg每次需要随机选择客户端，而FedPer需要激活所有客户端。
2. FedAvg需要聚合所有参数，FedPer只是聚合基础层参数。
3. FedAvg中只有一个全局模型，所有客户端都使用该全局模型，而FedPer中每个客户端都有自己的个性化模型。

## 联邦学习配置
config.py文件
关键参数配置
```
    # client
    # 客户端总数
    'client.amount': 10,
    # 本地迭代次数
    'local_epoch': 2,
    # 是否开启本地模型评估
    'local_OpenEval': False,

    # server
    # 全局迭代次数
    'gobal_epoch': 20,
    # 是否开启模型评估
    'openEval': True,

    # 数据集(cifar,mnist)
    'dataset': 'cifar',
    # 学习率
    'learn_rate': 0.01,
    # 数据集批次
    'BATCH_SIZE': 64,
```
本地联邦学习模拟采用串行和随机分配训练集的方式，进行多端环境模拟
```python
    # 创建一个服务
    sr = Server()
    model, version = sr.get_model()
    # 加入工作节点
    for i in range(config.my_conf['client.amount']):
        # 注册客户端，向每个节点分发模型
        sr.addClient(Client(model=model, mod_version=version, lr=config.my_conf['learn_rate'], client_id=i),
                    '127.0.0.1:808{}'.format(i))
    # 开始全局迭代训练
    for i in range(config.my_conf['gobal_epoch']):
        sr.start_train()
        # 保存模型
        # sr.saveModel('data/model/gobal/{}/network_{}_{}.pth'.format(config.my_conf['test_mod'], config.my_conf['test_mod'], i))
```
节点启动
```linux
# 在8080启动第一个SN节点,入口节点(使用本机ip 127.0.0.1:8080)
python main.py --port 8080 --nt SN --entry 222.197.211.74:8080 -z
# 在8081启动SN节点(节点会向入口地址注册)
python main.py --port 8081 --nt SN --entry 222.197.211.74:8080
# 在8082启动EN节点，EN节点会向入口地址查询合适的SN节点，并向其注册
python main.py --port 8082 --nt EN --entry 222.197.211.74:8080
# 控制台，需要一个SN节点的地址，不能接入EN节点
python Console.py --ip 222.197.211.74 --port 8080
```
## 控制台命令

|       命令       | 含义      |
|:--------------:|:--------|
|    train()     | 开始模型训练  |
|    pause()     | 暂停网络    |
|     save()     | 保存训练模型  |
|     eval()     | 评估模型    |
|    getSN()     | 获取SN列表  |
|   getModel()   | 下载服务端模型 |
| quit(),exit()  | 退出控制台   |
|   trainFL()    | 启动fl训练  |

M-SN(222.197.211.58:8080) <br>
&emsp;&emsp;|----SN(222.197.211.58:8081)<br>
&emsp;&emsp;&emsp;&emsp;|----EN(222.197.211.58:8083)<br>
&emsp;&emsp;&emsp;&emsp;|----EN(222.197.211.58:8084)