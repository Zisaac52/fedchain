# 联邦学习与区块链
环境：python 3.7
## 区块链仿真
使用grpc框架实现rpc通信，建立区块链节点
由于资源有限，利用mysql模拟区块存储

必要的组件库
```
pip install -r requirements.txt
```

proto文件编译命令，在proto文件所在位置运行该命令
```
python -m grpc_tools.protoc -I. --python_out=./base_package --grpc_python_out=./base_package ./data.proto
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
