import argparse

from blockchain.node.console import console_cmd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='console')
    # param是参数的名字，type是要传入参数的数据类型，help是该参数的提示信息
    # 端口
    parser.add_argument('--ip', required=True, type=str, help='eg. 127.0.0.1')
    parser.add_argument('--port', required=True, type=str, help='eg. 8080')
    # 获得传入的参数
    args = parser.parse_args()
    console_cmd({'ip': args.ip, 'port': args.port})
