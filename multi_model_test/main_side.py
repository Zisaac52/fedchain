from torchvision.models import resnet18


def random():
    return 'None'


def runtest_result():
    pass


if __name__ == '__main__':
    a = {
        'a': 1,
        'b': 2
    }
    print(a.get('c', random)())

