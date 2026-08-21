#!/usr/bin/env python3
"""
纯训练脚本 - 无 UI，命令行运行

用法:
    python scripts/train.py --epochs 100 --batch-size 64
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional


async def run_training(
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_agents: int = 8,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False
):
    """
    运行训练循环
    
    Args:
        num_epochs: 训练轮次
        batch_size: batch size
        learning_rate: 学习率
        num_agents: 智能体数量
        checkpoint_path: 检查点路径
        verbose: 详细日志
    """
    print("初始化训练环境...")
    
    # 导入依赖
    from config.hot_reload import ConfigManager
    from model import AlphaZeroResNet, ReplayBuffer, Trainer, TrainingConfig
    from agent import ModelAgent, TeacherAgent
    from engine.scheduler import GameScheduler
    from engine.scoring import ScoreManager
    
    # 加载配置
    config = ConfigManager()
    config.update({
        'N_AGENTS': num_agents,
        'BATCH_SIZE': batch_size,
        'LR': learning_rate,
        'REPLAY_BUFFER_SIZE': 500,
        'MCTS_ITERATIONS': 50,
        'CONCURRENT_GAMES': min(4, num_agents // 2)
    })
    
    # 创建模型池
    print(f"创建 {num_agents} 个智能体...")
    agents = []
    for i in range(num_agents):
        model = AlphaZeroResNet(num_res_blocks=8, channels=128)
        
        # 前 50% 为 Teacher，后 50% 为 Student
        is_teacher = i < num_agents // 2
        
        if is_teacher:
            agent = TeacherAgent(
                agent_id=f"agent_{i}",
                model=model,
                use_mcts=False
            )
        else:
            agent = ModelAgent(
                agent_id=f"agent_{i}",
                model=model,
                is_teacher=False,
                mcts_iterations=50
            )
        
        agents.append(agent)
    
    # 加载检查点
    if checkpoint_path:
        print(f"加载检查点：{checkpoint_path}")
        from model import load_checkpoint
        # 加载到所有学生智能体
        for agent in agents:
            if not agent.is_teacher:
                try:
                    load_checkpoint(checkpoint_path, agent.model)
                except FileNotFoundError:
                    print(f"警告：无法加载检查点 {checkpoint_path}")
                    break
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=config.get('REPLAY_BUFFER_SIZE', 500))
    
    # 创建训练器
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        alpha_distill=config.get('ALPHA', 1.0),
        beta_selfplay=config.get('BETA', 0.5),
        gamma_l2=config.get('GAMMA', 0.01)
    )
    
    trainer = Trainer(
        agents=[a for a in agents if not a.is_teacher],  # 只训练 Student
        replay_buffer=replay_buffer,
        config=training_config
    )
    
    # 创建调度器和计分管理器
    scheduler = GameScheduler(agents, max_concurrent=config.get('CONCURRENT_GAMES', 4))
    score_manager = ScoreManager(window_size=config.get('WINDOW_X', 10))
    
    # 训练循环
    print(f"\n开始训练 - {num_epochs} 轮")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 1. 匹配对局
        matchups = scheduler.create_matchups()
        
        # 2. 执行对局
        games_completed = 0
        for game in matchups:
            result = await scheduler.play_game(game)
            
            if result:
                games_completed += 1
                
                # 记录结果
                score_manager.record_result(
                    game_id=result['game_id'],
                    white_id=result['white_id'],
                    black_id=result['black_id'],
                    result=result['result']
                )
                
                # 添加到回放缓冲区
                if result.get('trajectory'):
                    replay_buffer.push(result['trajectory'])
        
        # 3. 更新身份（从第 Y 局起）
        if epoch >= config.get('START_Y', 5):
            score_manager.update_identities(agents)
        
        # 4. 训练
        if len(replay_buffer) >= batch_size:
            loss_info = await trainer.train_step()
            
            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Games: {games_completed:2d} | "
                      f"Loss: {loss_info['total']:.4f} | "
                      f"Distill: {loss_info['distill']:.4f} | "
                      f"SelfPlay: {loss_info['selfplay']:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {loss_info['total']:.4f}")
        
        epoch_time = time.time() - epoch_start
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_dir = Path("logs/training")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存最佳学生模型
            best_student = max(
                [a for a in agents if not a.is_teacher],
                key=lambda x: x.score
            )
            
            from model import save_checkpoint
            save_checkpoint(
                model=best_student.model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                score=best_student.score,
                save_dir=str(checkpoint_dir)
            )
            
            print(f"已保存检查点 (Epoch {epoch})")
    
    total_time = time.time() - start_time
    
    # 打印总结
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"总时间：{total_time:.1f}s ({total_time/num_epochs:.2f}s/轮)")
    print("\n最终排名:")
    
    sorted_agents = sorted(agents, key=lambda x: x.score, reverse=True)
    for i, agent in enumerate(sorted_agents[:10], 1):
        role = "Teacher" if agent.is_teacher else "Student"
        print(f"  {i:2d}. {agent.agent_id:15s} | Score: {agent.score:.3f} | "
              f"Role: {role:7s} | Win Rate: {agent.win_rate:.1%}")
    
    return {
        'epochs_completed': num_epochs,
        'total_time': total_time,
        'final_scores': {a.agent_id: a.score for a in agents}
    }


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='ChessRL 训练脚本')
    
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--agents', type=int, default=8, help='智能体数量')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(run_training(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_agents=args.agents,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose
        ))
        
        print("\n训练成功完成!")
        
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n错误：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
