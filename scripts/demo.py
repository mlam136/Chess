#!/usr/bin/env python3
"""
演示模式脚本 - 可视化自动对局

用法:
    python scripts/demo.py --agents 8 --speed 2x
"""

import argparse
import asyncio
from typing import Optional


async def run_demo(
    num_agents: int = 8,
    speed_multiplier: float = 1.0,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False
):
    """
    运行演示
    
    Args:
        num_agents: 智能体数量
        speed_multiplier: 速度倍数（1x, 2x, 5x, max）
        checkpoint_path: 检查点路径
        verbose: 详细日志
    """
    print("=" * 60)
    print("ChessRL - 演示模式")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 智能体数量：{num_agents}")
    print(f"  - 速度：{speed_multiplier}x")
    if checkpoint_path:
        print(f"  - 检查点：{checkpoint_path}")
    print()
    
    # 导入依赖
    from config.hot_reload import ConfigManager
    from model import AlphaZeroResNet, load_checkpoint
    from agent import ModelAgent, TeacherAgent, RandomAgent
    from engine.scheduler import GameScheduler
    from engine.scoring import ScoreManager
    from viz.app import ChessRLApp
    
    # 加载配置
    config = ConfigManager()
    config.update({
        'N_AGENTS': num_agents,
        'CONCURRENT_GAMES': min(4, num_agents // 2),
        'MCTS_ITERATIONS': 30,  # 演示时使用较少迭代以加快速度
        'TIMEOUT_T': 30
    })
    
    # 创建智能体
    print(f"创建 {num_agents} 个智能体...")
    agents = []
    
    for i in range(num_agents):
        # 演示模式混合使用模型和随机智能体
        if i < 2:
            # 前两个为随机智能体（保证有基本对局）
            agent = RandomAgent(agent_id=f"random_{i}", delay=0.1)
        else:
            # 其他为模型智能体
            model = AlphaZeroResNet(num_res_blocks=8, channels=128)
            
            # 如果有检查点则加载
            if checkpoint_path:
                try:
                    load_checkpoint(checkpoint_path, model)
                except FileNotFoundError:
                    print(f"警告：无法加载检查点 {checkpoint_path}，使用随机初始化模型")
            
            is_teacher = i < num_agents // 2
            
            if is_teacher:
                agent = TeacherAgent(
                    agent_id=f"teacher_{i}",
                    model=model,
                    use_mcts=False,
                    temperature=0.5
                )
            else:
                agent = ModelAgent(
                    agent_id=f"student_{i}",
                    model=model,
                    is_teacher=False,
                    mcts_iterations=30,
                    temperature=0.8
                )
        
        agents.append(agent)
    
    # 计算走子延迟（根据速度调整）
    base_delay = 0.3
    think_delay = base_delay / speed_multiplier if speed_multiplier < 999 else 0.01
    
    for agent in agents:
        if hasattr(agent, 'think_delay'):
            agent.think_delay = think_delay
    
    # 创建调度器和计分管理器
    scheduler = GameScheduler(agents, max_concurrent=config.get('CONCURRENT_GAMES', 4))
    score_manager = ScoreManager(window_size=config.get('WINDOW_X', 10))
    
    # 创建可视化应用
    app = ChessRLApp(
        agents=agents,
        scheduler=scheduler,
        score_manager=score_manager,
        config=config,
        verbose=verbose
    )
    
    print("\n启动演示窗口...")
    print("控制说明:")
    print("  - ESC / Q: 退出")
    print("  - P: 暂停/继续")
    print("  - +: 加速")
    print("  - -: 减速")
    print("  - S: 保存截图")
    print("=" * 60)
    
    # 运行演示
    await app.run_demo(speed_multiplier=speed_multiplier)
    
    # 打印最终结果
    print("\n演示结束!")
    print("\n最终排名:")
    
    sorted_agents = sorted(agents, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    for i, agent in enumerate(sorted_agents[:10], 1):
        score = agent.score if hasattr(agent, 'score') else 0
        win_rate = agent.win_rate if hasattr(agent, 'win_rate') else 0
        print(f"  {i:2d}. {agent.agent_id:15s} | Score: {score:.3f} | Win Rate: {win_rate:.1%}")
    
    return {
        'agents': len(agents),
        'final_scores': {a.agent_id: (a.score if hasattr(a, 'score') else 0) for a in agents}
    }


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='ChessRL 演示脚本')
    
    parser.add_argument('--agents', type=int, default=8, help='智能体数量')
    parser.add_argument('--speed', type=str, default='1x', 
                        choices=['1x', '2x', '5x', 'max'], help='演示速度')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    # 解析速度
    speed_map = {'1x': 1.0, '2x': 2.0, '5x': 5.0, 'max': 999}
    speed_multiplier = speed_map.get(args.speed, 1.0)
    
    try:
        result = asyncio.run(run_demo(
            num_agents=args.agents,
            speed_multiplier=speed_multiplier,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose
        ))
        
        print("\n演示成功完成!")
        
    except KeyboardInterrupt:
        print("\n演示被中断")
    except Exception as e:
        print(f"\n错误：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
