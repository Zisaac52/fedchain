import os
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
    'momentum': 0.001,
    'BATCH_SIZE': 64,

    # ------------------------------------------------------------------
    # Server/global training knobs
    'gobal_epoch': 101,
    'openEval': True,
    'learn_rate': 0.01,
    # 数据集选项：mnist | fmnist | cifar | cifar100
    'dataset': 'mnist',
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
    'scheduler_mode': 'ddmlts_b',
    'ddmlts_alpha': 1.0,
    'ddmlts_cluster_count': 2,
    'ddmlts_mini_batch': 8,
    'ddmlts_tau_ratio': 0.25,
    'state_vector_weights': (1 / 3, 1 / 3, 1 / 3),

    # Staleness weighting (for async aggregation)
    'staleness_mode': 'reciprocal',  # reciprocal | polynomial | exponential | constant
    'staleness_power': 0.5,          # used in polynomial
    'staleness_lambda': 0.5,         # used in exponential

    # Metrics toggles (for reviewer requests)
    'enable_precision_metrics': True,
    'enable_mae': False,
    'enable_rrmse': False,
    'enable_r2': False,
}

_state_weight_env = os.environ.get('STATE_VECTOR_WEIGHTS')
if _state_weight_env:
    try:
        weights = tuple(
            float(item.strip())
            for item in _state_weight_env.split(',')
            if item.strip() != ''
        )
        if weights:
            my_conf['state_vector_weights'] = weights
    except ValueError:
        pass

_staleness_mode = os.environ.get('STALENESS_MODE')
if _staleness_mode:
    my_conf['staleness_mode'] = _staleness_mode.lower()

_staleness_power = os.environ.get('STALENESS_POWER')
if _staleness_power:
    try:
        my_conf['staleness_power'] = float(_staleness_power)
    except ValueError:
        pass

_staleness_lambda = os.environ.get('STALENESS_LAMBDA')
if _staleness_lambda:
    try:
        my_conf['staleness_lambda'] = float(_staleness_lambda)
    except ValueError:
        pass

# Backwards-compatibility aliases so legacy modules (README snippets,
# my_test.py, etc.) keep working until they are updated.
my_conf['client.amount'] = my_conf['client_amount']
