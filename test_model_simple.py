import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 按项目的模型结构导入
# fl/model.py 里面应该有 mnist_Net 这个类，如果名字不一样，改成对应的类名即可
from fl.model import mnist_Net  # 如果报错，再去 fl/model.py 里看实际类名

def evaluate_one_checkpoint(path: str, device: torch.device):
    print(f"\n=== Evaluating {path} ===")

    # 2. 构建模型并加载参数
    model = mnist_Net().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3. 准备 MNIST 测试集（和训练时同一个数据）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root="../data/MNIST",
        train=False,
        download=True,      # 之前已经下载过了
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    print(f"Test accuracy = {acc:.4f}, loss = {avg_loss:.4f}")
    return acc, avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 你可以只看最后一轮，也可以循环所有轮次
    for i in range(0, 20):  # 对应 gmodel-0.pth ~ gmodel-19.pth
        path = f"data/flm/gmodel-{i}.pth"
        try:
            evaluate_one_checkpoint(path, device)
        except FileNotFoundError:
            print(f"File not found: {path}, 跳过")
        except Exception as e:
            print(f"评估 {path} 时出错: {e}")

if __name__ == "__main__":
    main()
