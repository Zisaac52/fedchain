# 评估模型
import torch

from fl.loadTrainData import load2MnistLoader


def evalmodel(model_sev, model_cln):
    _, testsets = load2MnistLoader()
    test_loader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=True)
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    model_sev.eval()
    model_cln.eval()
    for data in test_loader:
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        dataset_size += inputs.size()[0]
        # 开始训练
        fx = model_cln(inputs)
        cln_fx = fx.clone().detach().requires_grad_(True)
        # 正向计算到服务端，输出损失和结果
        output = model_sev(cln_fx)
        total_loss += torch.nn.functional.cross_entropy(output, targets).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        # 按照从左往右的 第一维 取出最大值的索引 torch.max()
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    acc = 100.0 * (float(correct) / float(dataset_size))  # 准确率
    aver_loss = total_loss / dataset_size  # 平均损失
    return acc, aver_loss
