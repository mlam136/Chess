#!/usr/bin/env python3
"""
ChessRL 主入口 - CLI 命令行工具

用法:
    python main.py --mode play     # 人机对弈模式
    python main.py --mode train    # 纯训练模式
    python main.py --mode demo     # 演示模式（可视化自动对局）
"""

import argparse
import sys
import asyncio
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ChessRL - 多智能体国际象棋自博弈学习平台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --mode play              # 启动人机对弈
    python main.py --mode train --epochs 10 # 训练 10 轮
    python main.py --mode demo --speed 2x   # 2 倍速演示
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['play', 'train', 'demo'],
        default='demo',
        help='运行模式：play(对弈), train(训练), demo(演示)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='配置文件路径'
    )
    
    # 训练相关参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮次数量'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='训练 batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学习率'
    )
    
    # 演示相关参数
    parser.add_argument(
        '--speed',
        type=str,
        default='1x',
        choices=['1x', '2x', '5x', 'max'],
        help='演示速度'
    )
    
    parser.add_argument(
        '--agents',
        type=int,
        default=8,
        help='智能体数量'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='加载的检查点路径'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    
    return parser.parse_args()


async def run_play_mode(args):
    """运行对弈模式"""
    print("=" * 60)
    print("ChessRL - 人机对弈模式")
    print("=" * 60)
    
    try:
        from viz.app import ChessRLApp
        from agent import create_human_agent, create_model_agent
        from model import create_model
        
        # 创建模型
        model = create_model(num_res_blocks=8, channels=128)
        
        # 加载检查点
        if args.checkpoint:
            from model import load_checkpoint
            load_checkpoint(args.checkpoint, model)
            print(f"已加载检查点：{args.checkpoint}")
        
        # 创建智能体
        human = create_human_agent("human_player")
        ai = create_model_agent("ai_opponent", model=model, is_teacher=False)
        
        # 启动应用
        app = ChessRLApp(
            agents=[human, ai],
            config_path=args.config,
            verbose=args.verbose
        )
        
        await app.run_human_vs_ai()
        
    except ImportError as e:
        print(f"错误：缺少依赖模块 - {e}")
        print("请确保已安装所有依赖：pip install -r requirements.txt")
        sys.exit(1)


async def run_train_mode(args):
    """运行训练模式"""
    print("=" * 60)
    print("ChessRL - 训练模式")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 轮次：{args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - 学习率：{args.lr}")
    print(f"  - 智能体数量：{args.agents}")
    print()
    
    try:
        from scripts.train import run_training
        
        await run_training(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_agents=args.agents,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose
        )
        
    except ImportError as e:
        print(f"错误：缺少依赖模块 - {e}")
        sys.exit(1)


async def run_demo_mode(args):
    """运行演示模式"""
    print("=" * 60)
    print("ChessRL - 演示模式")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 速度：{args.speed}")
    print(f"  - 智能体数量：{args.agents}")
    print()
    
    try:
        from scripts.demo import run_demo
        
        speed_map = {'1x': 1.0, '2x': 2.0, '5x': 5.0, 'max': 999}
        speed_multiplier = speed_map.get(args.speed, 1.0)
        
        await run_demo(
            num_agents=args.agents,
            speed_multiplier=speed_multiplier,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose
        )
        
    except ImportError as e:
        print(f"错误：缺少依赖模块 - {e}")
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # 根据模式运行
    try:
        if args.mode == 'play':
            asyncio.run(run_play_mode(args))
        elif args.mode == 'train':
            asyncio.run(run_train_mode(args))
        elif args.mode == 'demo':
            asyncio.run(run_demo_mode(args))
        else:
            print(f"未知模式：{args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n程序已中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
