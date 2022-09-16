import torch.cuda
from torch import nn

from fl.loadTrainData import load2Loader
from fl.model import mnist_Net

# 2. 设置超参数
BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 加载数据
    train_loader, test_loader = load2Loader(BATCH_SIZE)
    model = mnist_Net()
    if torch.cuda.is_available():
        model.cuda()
    # 建立损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn.cuda()

    # 建立优化器
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置训练次数
    total_train_step = 0

    # 设置测试次数
    total_test_step = 0

    # 训练的次数
    epoch = 10


    for i in range(epoch):
        print(f"------------第{i + 1}轮训练-------------")

        model.train()
        for data in train_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = model(imgs)
            loss = loss_fn(output, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f"训练次数：{total_train_step}  loss：{loss.item()}")

        # 保存参数
        torch.save(model, "./data/model/network_{}.pth".format(i))
        print("模型已保存")
