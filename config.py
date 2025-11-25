import torch

# Base configuration shared by both the standalone FL runner (fl/main.py)
# and the blockchain nodes when they launch FL/console experiments.
my_conf = {
    # ------------------------------------------------------------------
    # Client-side knobs
    'client_amount': 10,
    'client_k': 4,
    'local_epoch': 4,
    'local_OpenEval': False,
    'optimizer': 'sgd',
    'momentum': 0.95,
    'BATCH_SIZE': 64,

    # ------------------------------------------------------------------
    # Server/global training knobs
    'gobal_epoch': 101,
    'openEval': True,
    'learn_rate': 0.01,
    # 数据集选项：mnist | fmnist | cifar | cifar100
    'dataset': 'cifar100',
    'load_model': False,
    'load_path': './data/model/gobal/network_{}_{}_{}_{}_{}_{}.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # ------------------------------------------------------------------
    # Straggler / synthetic-delay experiments
    'issyntest': True,
    'test_client_id': [4],
    'isTest': False,
    'test_mod': 'os',  # ns | os
    'test_in_nepoch': 4,
    'test_param_xi': 0.05,

    # ------------------------------------------------------------------
    # Scheduler selection + DDMLTS parameters
    # fedavg | ddmlts_a | ddmlts_b | bafl | perfeds2
    'scheduler_mode': 'perfeds2',
    'ddmlts_alpha': 1.0,
    'ddmlts_cluster_count': 2,
    'ddmlts_mini_batch': 8,
    'ddmlts_tau_ratio': 0.25,

    # Metrics toggles (for reviewer requests)
    'enable_precision_metrics': True,
    'enable_mae': False,
    'enable_rrmse': False,
    'enable_r2': False,
}

# Backwards-compatibility aliases so legacy modules (README snippets,
# my_test.py, etc.) keep working until they are updated.
my_conf['client.amount'] = my_conf['client_amount']
