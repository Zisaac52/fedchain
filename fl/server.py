import copy
import logging
import math
import os
import random
import sys

import torch
import torchvision

from model import mnist_Net, FmCNN, MyResNet18
from modelEval import model_eval

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from blockchain.node.service.ddmlts_scheduler import DDMLTSScheduler
except ImportError:
    DDMLTSScheduler = None

logger = logging.getLogger()


# 服务器端重复进行客户端采样、参数分发、参数聚合三个步骤。其中参数聚合和参数分发都只针对基础层。
class Server:
    def __init__(self, conf):
        self.config = conf
        self._gobal_model = None
        self._diffval = None
        self.version = 10
        self._clientList = []
        self._clientlab = []
        # 第一次运行，初始化并分发模型
        self._initModel()
        # 对当前训练计数，用于识别当前学习状态
        self._count = 0
        self._metric_header_logged = False
        self.scheduler = None
        self._client_stats = {}
        self._active_plan = {}
        self._round_client_count = 0
        scheduler_mode = getattr(self.config, 'scheduler_mode', 'fedavg').lower()
        if DDMLTSScheduler is not None and scheduler_mode in ('ddmlts_a', 'ddmlts_b'):
            try:
                self.scheduler = DDMLTSScheduler(dict(self.config))
                logger.info('DDMLTS scheduler enabled: mode=%s', scheduler_mode)
            except Exception as err:
                logger.warning('Failed to init DDMLTS scheduler: %s', err)

    def _initModel(self):
        if self._gobal_model is None:
            dataset_name = self.config.dataset.lower()
            if dataset_name == "cifar":
                if self.config.load_model:
                    self._gobal_model = torch.load(self.config.load_path)
                else:
                    self._gobal_model = MyResNet18(num_classes=10)
            elif dataset_name == "cifar100":
                self._gobal_model = MyResNet18(num_classes=100)
            elif dataset_name == "mnist":
                # self.gobal_model = models.mnasnet1_0()
                self._gobal_model = mnist_Net()
                # 使用预训练的模型
                # self._gobal_model = torch.load("data/model/base/network_base.pth")
            elif dataset_name == "fmnist":
                self._gobal_model = FmCNN()
            else:
                raise ValueError("config.my_conf.dataset配置有误，无法找到该项！")
            self._gobal_model.to(self.config.device)

    # 统一分发模型
    # 循环遍历发送初始模型到client
    def dispatch(self, cln):
        cln.setModelFromServer(self._gobal_model, self.version)

    # 对diff以权重u进行累加
    def accumulator(self, diff, u_w):
        if self._diffval is None:
            # 深拷贝，防止意外更改
            self._diffval = copy.deepcopy(diff)
        else:
            for name, value in self._diffval.items():
                add_per_layer = diff[name] * u_w
                add_per_layer = add_per_layer.to(self.config.device)
                value = value.float()
                add_per_layer = add_per_layer.float()
                value.add_(add_per_layer)
        if self._round_client_count is None:
            self._round_client_count = 0
        self._round_client_count += 1

    # 进行模型聚合，接收来自worknode的diff，更新至自己的模型
    def aggregation(self):
        if self._diffval is not None:
            participating = max(1, self._round_client_count)
            avg_w = 1 / participating
            # 全局模型参数更新,得到新的global_model
            for name, value in self._gobal_model.state_dict().items():
                update_per_layer = self._diffval[name]
                update_per_layer = update_per_layer.to(self.config.device)
                # 计算平均diff，将全部客户端的结果进行聚合
                update_per_layer = update_per_layer * avg_w
                # 判断是否类型相同，不同则转换
                if value.type() != update_per_layer.type():
                    value.add_(update_per_layer.to(torch.int64))
                else:
                    value.add_(update_per_layer)
                # value = value.float()
                # update_per_layer = update_per_layer.float()
                # value.add_(update_per_layer)
            # 聚合完成后将全局diff清空
            self._diffval = None
        self._round_client_count = 0

    # 保存每次聚合完成的模型
    def saveModel(self, path):
        torch.save(self._gobal_model, path)
        logger.info('保存模型，路径：{}'.format(path))

    def addClient(self, client, cln_name):
        self._clientList.append(client)
        self._clientlab.append(cln_name)
        self._client_stats[str(client.get_client_id())] = {
            'dataset_size': client.get_dataset_size(),
            'last_elapsed': 1.0,
            'speed': client.estimate_speed()
        }
        logger.info('节点-{}-加入'.format(cln_name))

    def random_client(self):
        '''
        客户端选择算法
        :return:
        '''
        lst = []
        rd = random.sample(range(0, self.config.client_amount), self.config.client_k)
        for cid in rd:
            lst.append(self._clientList[cid])
        return lst

    # 原始的训练聚合方法
    def start_train(self):
        if self.config.openEval:
            metrics = model_eval(self._gobal_model, self.config.device)
            self._log_metrics(self._count, metrics)
        if self._should_use_scheduler():
            self._apply_scheduler_plan()
        self._round_client_count = 0
        self._count += 1
        max_client_time = 0.0
        for i, cln in enumerate(self.random_client()):
            # 如果开启了测试,且配置的滞后客户端不为0，则判断当前client是否为滞后节点,是则跳过该节点
            if self.config.issyntest and len(self.config.test_client_id) != 0:
                if cln.get_client_id() in self.config.test_client_id:
                    continue
            diff, ver, elapsed = cln.local_train()
            max_client_time = max(max_client_time, elapsed)
            self._update_client_stats(cln, elapsed)
            if ver != -1:
                delta = max(0, self.version - ver)
                weight = self._staleness_weight(delta)
                self.accumulator(diff, weight)
        # 模型聚合
        self.aggregation()
        self.version += 1
        logger.info('epoch {} C_gmax(sec) {}'.format(self._count - 1, max_client_time))
        self.dispatch_normal()

    # 获取初始模型
    def get_model(self):
        return self._gobal_model, self.version

    def dispatch_normal(self):
        for cln in self._clientList:
            self.dispatch(cln)

    def _should_use_scheduler(self):
        scheduler_mode = getattr(self.config, 'scheduler_mode', 'fedavg').lower()
        return self.scheduler is not None and scheduler_mode in ('ddmlts_a', 'ddmlts_b')

    def _build_state_vectors(self):
        vectors = {}
        for cln in self._clientList:
            cid = str(cln.get_client_id())
            stats = self._client_stats.get(cid)
            if stats:
                dataset = float(stats.get('dataset_size', cln.get_dataset_size() or 1.0))
                elapsed = float(stats.get('last_elapsed', 1.0))
                speed = float(stats.get('speed', cln.estimate_speed()))
                vectors[cid] = (dataset, max(elapsed, 1e-6), speed, speed)
            else:
                vectors[cid] = cln.get_state_vector()
        return vectors

    def _apply_scheduler_plan(self):
        if not self._should_use_scheduler():
            return
        vectors = self._build_state_vectors()
        if not vectors:
            return
        try:
            plan = self.scheduler.build_plan(vectors)
        except Exception as err:
            logger.warning('scheduler error: %s', err)
            return
        self._active_plan = plan
        summary = {cid: {'epoch': data.get('local_epoch'), 'micro': data.get('micro_batches')}
                   for cid, data in plan.items()}
        if summary:
            logger.info('DDMLTS assignment: %s', summary)
        for cln in self._clientList:
            assignment = plan.get(str(cln.get_client_id()))
            cln.apply_workload(assignment)

    def _update_client_stats(self, client, elapsed):
        cid = str(client.get_client_id())
        self._client_stats[cid] = {
            'dataset_size': client.get_dataset_size(),
            'last_elapsed': elapsed,
            'speed': client.estimate_speed()
        }

    def _staleness_weight(self, delta):
        mode = getattr(self.config, 'staleness_mode', 'reciprocal')
        mode = mode.lower() if isinstance(mode, str) else 'reciprocal'
        delta = max(0.0, float(delta))
        if mode == 'constant':
            return 1.0
        if mode == 'exponential':
            lam = max(0.0, float(getattr(self.config, 'staleness_lambda', 0.5)))
            return math.exp(-lam * delta)
        if mode == 'polynomial':
            power = max(0.0, float(getattr(self.config, 'staleness_power', 0.5)))
            return 1.0 / ((delta + 1.0) ** (power if power > 0 else 1.0))
        # reciprocal default
        return 1.0 / (delta + 1.0)

    def _build_metric_headers(self):
        headers = ['gobal_epoch', 'Accuracy', 'loss', 'Precision', 'Recall', 'F1-score']
        if getattr(self.config, 'enable_mae', False):
            headers.append('MAE')
        if getattr(self.config, 'enable_rrmse', False):
            headers.append('RRMSE')
        if getattr(self.config, 'enable_r2', False):
            headers.append('R-squared')
        return headers

    def _format_metric_row(self, epoch_label, metrics):
        row = [
            str(epoch_label),
            f"{metrics.get('accuracy', 0.0):.2f}",
            f"{metrics.get('loss', 0.0):.6f}",
            f"{metrics.get('precision', 0.0):.2f}",
            f"{metrics.get('recall', 0.0):.2f}",
            f"{metrics.get('f1', 0.0):.2f}",
        ]
        if getattr(self.config, 'enable_mae', False):
            row.append(f"{metrics.get('mae', 0.0):.6f}")
        if getattr(self.config, 'enable_rrmse', False):
            row.append(f"{metrics.get('rrmse', 0.0):.6f}")
        if getattr(self.config, 'enable_r2', False):
            row.append(f"{metrics.get('r2', 0.0):.6f}")
        return ','.join(row)

    def _log_metrics(self, epoch_idx, metrics):
        if not self._metric_header_logged:
            logger.info(','.join(self._build_metric_headers()))
            self._metric_header_logged = True
        logger.info(self._format_metric_row(epoch_idx, metrics))

    # -----------------------------------------------------------------------------------?
    # -----------------------------------------------------------------------------------?
    # -----------------------------------------------------------------------------------?
    # 测试聚合方法，用于平衡测试的，忽略以下方法train，test_my_scheme，dispatch_test
    def train(self, currentEpoch):

        if self.config.openEval:
            metrics = model_eval(self._gobal_model, self.config.device)
            row = self._format_metric_row(self._count, metrics)
            print(row)
        self._count += 1
        self._round_client_count = 0

        if self._should_use_scheduler():
            self._apply_scheduler_plan()
        # 训练聚合
        max_client_time = 0.0
        for i, cln in enumerate(self._clientList):
            skip_flag = False
            u = 1
            if self.config.isTest:
                skip_flag = self.test_my_scheme(i, currentEpoch)
                if skip_flag:
                    continue
                else:
                    diff, ver, elapsed = cln.local_train()
                    max_client_time = max(max_client_time, elapsed)
                    self._update_client_stats(cln, elapsed)
                # 全局模型分发错误，大部分节点都落后一个版本
                # 若是新方案则进行更改u的值
                # print('第{}轮，客户端{}版本差异：{}'.format(currentEpoch, i, ver != self.version))
                if self.config.test_mod.lower() == 'ns':
                    if ver != self.version:
                        u = self.config.test_param_xi / (self.version - ver)
                    self.accumulator(diff, u)
                    # print('全局第{}次迭代，当前全局模型版本{}，client-{}与全局版本差距{}'.format(currentEpoch, self.version, i, self.version - ver))
                if self.config.test_mod.lower() == 'os':
                    if not skip_flag:
                        self.accumulator(diff, u)
            else:
                diff, ver, elapsed = cln.local_train()
                max_client_time = max(max_client_time, elapsed)
                self._update_client_stats(cln, elapsed)
                if ver != -1:
                    delta = max(0, self.version - ver)
                    weight = self._staleness_weight(delta)
                else:
                    weight = u
                self.accumulator(diff, weight)

        # 模型聚合
        self.aggregation()
        self.version += 1
        logger.info('epoch {} C_gmax(sec) {}'.format(self._count - 1, max_client_time))
        if self.config.isTest:
            # 在间隔0,3,6,9的时候分发模型
            self.dispatch_test(currentEpoch)
        else:
            self.dispatch_normal()

    def test_my_scheme(self, i, currentEpoch):
        skip_flag = False
        # 训练聚合
        u = 1
        if currentEpoch % self.config.test_in_nepoch != 0:
            # 跳过配置的客户端
            for j in self.config.test_client_id:
                # 在 test_in_nepoch 代训练 0, 3, 6, 9
                if i == j:
                    # 跳出循环，忽略该客户端本次更新
                    return True
        return skip_flag

    def dispatch_test(self, currentEpoch):
        # 对客户端进行筛选，其他正常客户端无需判断
        cln_id = [x for x in range(len(self._clientList))]
        cln_otr = list(set(cln_id) ^ set(self.config.test_client_id))
        for clnid in cln_otr:
            self.dispatch(self._clientList[clnid])
        if currentEpoch % self.config.test_in_nepoch == 0:
            for ids in self.config.test_client_id:
                self.dispatch(self._clientList[ids])
        # 用聚合的模型进行评估
        for ids in self.config.test_client_id:
            metrics = model_eval(
                self._gobal_model,
                self.config.device,
                testLoader=self._clientList[ids].getDataset()
            )
            row = self._format_metric_row(f'client-{ids}', metrics)
            print(f'节点id{ids},{row}')
