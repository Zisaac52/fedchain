import torch

try:
    from Configurator import Configurator
    from loadTrainData import load2MnistLoader, load_fashion_mnist, load2Cifar10Loader, load2Cifar100Loader
except ImportError:
    from fl.Configurator import Configurator
    from fl.loadTrainData import load2MnistLoader, load_fashion_mnist, load2Cifar10Loader, load2Cifar100Loader

config = Configurator().get_config()
device = config.device


def get_test_loader():
    if config.dataset.lower() == "mnist":
        mydatasets = load2MnistLoader(is_train=False)
    elif config.dataset.lower() == "fmnist":
        mydatasets = load_fashion_mnist(is_train=False)
    elif config.dataset.lower() == "cifar":
        mydatasets = load2Cifar10Loader(is_train=False)
    elif config.dataset.lower() == "cifar100":
        mydatasets = load2Cifar100Loader(is_train=False)
    else:
        raise Exception("only support mnist，cifar and fmnist!")
    return torch.utils.data.DataLoader(mydatasets, batch_size=config.BATCH_SIZE, shuffle=True)


test_loaders = get_test_loader()


def _compute_classification_metrics(conf_matrix):
    eps = 1e-12
    tp = conf_matrix.diag().float()
    fp = conf_matrix.sum(dim=0).float() - tp
    fn = conf_matrix.sum(dim=1).float() - tp

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    precision = precision_per_class.mean().item() * 100.0
    recall = recall_per_class.mean().item() * 100.0
    f1 = f1_per_class.mean().item() * 100.0
    return precision, recall, f1


def _compute_regression_metrics(preds, targets):
    eps = 1e-12
    pred_tensor = torch.tensor(preds, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    err = pred_tensor - target_tensor
    mae = torch.mean(torch.abs(err)).item()
    mse = torch.mean(err ** 2).item()
    rmse = torch.sqrt(err.pow(2).mean() + eps).item()
    mean_target = torch.mean(target_tensor)
    sst = torch.sum((target_tensor - mean_target) ** 2).item() + eps
    sse = torch.sum(err ** 2).item()
    r2 = 1.0 - (sse / sst)
    rrmse = rmse / (torch.mean(torch.abs(target_tensor)).item() + eps)
    return mae, rrmse, r2


def model_eval(model, device, testLoader=None):
    """用于评估模型指标"""
    model.eval()
    if testLoader is None:
        testLoader = test_loaders
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    conf_matrix = None
    need_extra = any([
        getattr(config, 'enable_mae', False),
        getattr(config, 'enable_rrmse', False),
        getattr(config, 'enable_r2', False)
    ])
    regression_preds = []
    regression_targets = []

    for batch_id, batch in enumerate(testLoader):
        inputs, target = batch
        inputs = inputs.to(device)
        target = target.to(device)
        dataset_size += inputs.size()[0]
        output = model(inputs)

        total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        if need_extra:
            regression_preds.extend(pred.detach().cpu().tolist())
            regression_targets.extend(target.detach().cpu().tolist())
        if conf_matrix is None:
            num_classes = output.shape[1]
            conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(target.view(-1).cpu(), pred.view(-1).cpu()):
            conf_matrix[t.long(), p.long()] += 1

    metrics = {
        'accuracy': 100.0 * (float(correct) / float(dataset_size)) if dataset_size else 0.0,
        'loss': total_loss / dataset_size if dataset_size else 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }

    if conf_matrix is not None:
        precision, recall, f1 = _compute_classification_metrics(conf_matrix)
        metrics.update({'precision': precision, 'recall': recall, 'f1': f1})

    if need_extra and len(regression_preds) == len(regression_targets) and len(regression_targets) > 0:
        mae, rrmse, r2 = _compute_regression_metrics(regression_preds, regression_targets)
        metrics['mae'] = mae
        metrics['rrmse'] = rrmse
        metrics['r2'] = r2

    return metrics


# 返回一个元组类型（正确的个数，总的测试集数量，准确率，损失值）
def model_eval_nograde(model, testset=None):
    testLoader = test_loaders if testset is None else testset
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
    return (100. * correct / len(testLoader.dataset)), (test_loss / len(testLoader.dataset))
    # return (
    #     correct,
    #     len(testLoader.dataset),
    #     100. * correct / len(testLoader.dataset),
    #     test_loss / len(testLoader.dataset)
    # )
