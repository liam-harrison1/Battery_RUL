"""
电池寿命预测项目统一命令行接口
支持训练、预测、实验管理功能
"""
import argparse
import json
import os
import glob
from datetime import datetime

def train_command(args):
    """执行训练命令"""
    from train_baselines import main as train_main
    train_main(args)

def predict_command(args):
    """执行预测命令"""
    from predict_baselines import main as predict_main
    # 确保传递config参数
    predict_main(args)

def list_experiments(args):
    """列出所有实验记录"""
    if not os.path.exists('experiments.json'):
        print("暂无实验记录")
        return
    
    with open('experiments.json', 'r') as f:
        experiments = json.load(f)
    
    print("\n实验记录列表:")
    print("-" * 80)
    print(f"{'ID':<5}{'时间':<20}{'模型':<10}{'验证MAE':<10}{'路径':<40}")
    print("-" * 80)
    for exp in experiments:
        print(f"{exp['id']:<5}{exp['timestamp']:<20}{exp['model']:<10}{exp.get('val_mae', 'N/A'):<10}{exp['path']:<40}")

def main():
    parser = argparse.ArgumentParser(
        description='电池寿命预测统一命令行工具',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(title='可用命令', dest='command')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练新模型')
    train_parser.add_argument('--model', required=True,
                             choices=['rnn', 'lstm', 'gru', 'transformer', 'saetr'],
                             help='模型类型')
    train_parser.add_argument('--config', default='configs/config.yaml',
                             help='配置文件路径 (默认: configs/config.yaml)')
    train_parser.set_defaults(func=train_command)

    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='使用训练好的模型进行预测')
    predict_parser.add_argument('--exp-id', type=int,
                               help='实验ID (使用list命令查看)')
    predict_parser.add_argument('--latest', action='store_true',
                               help='使用最新实验进行预测')
    predict_parser.add_argument('--config', type=str, default='configs/config.yaml',
                               help='配置文件路径 (默认: configs/config.yaml)')
    predict_parser.set_defaults(func=predict_command)

    # 实验管理命令
    list_parser = subparsers.add_parser('list', help='列出所有实验记录')
    list_parser.set_defaults(func=list_experiments)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()