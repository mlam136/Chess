#!/usr/bin/env python3
"""
纯训练脚本 - 支持 GUI 可视化监控

用法:
    python scripts/train.py --epochs 100 --batch-size 64           # 无可视化
    python scripts/train.py --epochs 100 --gui                     # 带 GUI 监控窗口
    python scripts/train.py --epochs 100 --gui --ascii             # GUI + ASCII 双模式
"""

import argparse
import asyncio
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Callable, Any


async def run_training(
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_agents: int = 8,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
    gui_viz: bool = False,
    ascii_viz: bool = False,
    viz_interval: int = 10
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
        gui_viz: 是否启用 GUI 可视化窗口
        ascii_viz: 是否启用 ASCII 终端显示
        viz_interval: 可视化间隔（多少轮显示一次）
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
    
    # 可视化设置
    update_queue = None
    gui_overlay = None
    gui_thread = None
    viz_callback = None
    
    if gui_viz:
        try:
            # 启动 GUI 线程
            update_queue = queue.Queue()
            gui_overlay, gui_thread = start_gui_thread(update_queue)
            viz_callback = create_gui_callback(update_queue, viz_interval)
            print(f"✓ GUI 可视化已启动")
        except ImportError as e:
            print(f"⚠ GUI 模块不可用：{e}")
            print("  回退到 ASCII 模式...")
            gui_viz = False
            ascii_viz = True
    
    if ascii_viz and not gui_viz:
        try:
            viz_callback = create_ascii_callback(viz_interval=viz_interval)
            print(f"✓ ASCII 可视化已启用 (每 {viz_interval} 轮显示一次)")
        except ImportError as e:
            print(f"⚠ ASCII 可视化模块不可用：{e}")
    
    # 训练循环
    print(f"\n开始训练 - {num_epochs} 轮")
    if gui_viz:
        print("模式：GUI 实时监控窗口")
    elif ascii_viz:
        print(f"模式：ASCII 终端显示 (间隔={viz_interval})")
    else:
        print("模式：无头训练 (仅日志)")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 更新 GUI 状态
        if gui_viz and update_queue:
            update_queue.put(("status", {"training": True, "epoch": epoch, "total_epochs": num_epochs}))
        
        # 1. 匹配对局
        matchups = scheduler.create_matchups()
        
        # 2. 执行对局
        games_completed = 0
        displayed_games = 0
        
        for game in matchups:
            # 检查是否需要可视化此局
            should_visualize = (
                (gui_viz or ascii_viz) and 
                viz_callback and 
                epoch % viz_interval == 0 and
                displayed_games == 0  # 每轮只显示一局
            )
            
            if should_visualize:
                result = await scheduler.play_game(game, callback=viz_callback)
                displayed_games += 1
            else:
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
        loss_info = None
        if len(replay_buffer) >= batch_size:
            loss_info = await trainer.train_step()
            
            # 更新 GUI 指标
            if gui_viz and update_queue and loss_info:
                update_queue.put(("metrics", {"loss": loss_info['total']}))
            
            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Games: {games_completed:2d} | "
                      f"Loss: {loss_info['total']:.4f} | "
                      f"Distill: {loss_info['distill']:.4f} | "
                      f"SelfPlay: {loss_info['selfplay']:.4f}")
            else:
                if epoch % 10 == 0 or loss_info:
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
    
    # 通知 GUI 训练结束
    if gui_viz and update_queue:
        update_queue.put(("stop", {}))
        # 等待 GUI 线程退出
        gui_thread.join(timeout=2.0)
    
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


def start_gui_thread(update_queue: queue.Queue):
    """
    在独立线程中启动 GUI 监控窗口
    
    Args:
        update_queue: 用于接收训练更新消息的队列
        
    Returns:
        (overlay, thread) - GUI 覆盖层对象和线程
    """
    import tkinter as tk
    from viz.training_overlay import TrainingOverlay
    
    # 创建 Tk 根窗口
    root = tk.Tk()
    root.withdraw()  # 先隐藏，等初始化完成再显示
    
    # 创建覆盖层
    overlay = TrainingOverlay(root, update_queue)
    
    # 显示窗口
    root.deiconify()
    
    # 创建 GUI 线程
    def gui_loop():
        try:
            root.mainloop()
        except Exception as e:
            print(f"GUI 错误：{e}")
    
    thread = threading.Thread(target=gui_loop, daemon=True)
    thread.start()
    
    return overlay, thread


def create_gui_callback(update_queue: queue.Queue, viz_interval: int = 10):
    """
    创建 GUI 可视化回调函数
    
    Args:
        update_queue: 消息队列
        viz_interval: 可视化间隔
        
    Returns:
        异步回调函数，接收 (board, move, info) 参数
    """
    import chess
    
    async def on_move(board: chess.Board, move: chess.Move, info: dict):
        """对局移动回调"""
        # 发送棋盘状态到 GUI
        board_state = {
            'fen': board.fen(),
            'last_move': move.uci(),
            'turn': 'white' if board.turn else 'black'
        }
        update_queue.put(("board", board_state))
        
        # 发送日志
        game_id = info.get('game_id', '?')
        move_num = info.get('move_number', 0)
        update_queue.put(("log", f"Game {game_id} | Move {move_num}: {move.uci()}"))
    
    return on_move


def create_ascii_callback(viz_interval: int = 10):
    """
    创建 ASCII 终端可视化回调函数
    
    Args:
        viz_interval: 可视化间隔
        
    Returns:
        异步回调函数，接收 (board, move, info) 参数
    """
    import chess
    
    async def on_move(board: chess.Board, move: chess.Move, info: dict):
        """对局移动回调"""
        game_id = info.get('game_id', '?')
        move_num = info.get('move_number', 0)
        white_name = info.get('white_name', 'White')
        black_name = info.get('black_name', 'Black')
        
        # 清屏并显示棋盘
        print("\n" + "=" * 60)
        print(f"Game: {game_id} | Move {move_num}: {white_name} vs {black_name}")
        print("=" * 60)
        
        # 渲染棋盘（ASCII）
        ascii_board = board.unicode()
        print(ascii_board)
        
        # 显示最后一步
        print(f"\nLast move: {move.uci()}")
        print(f"FEN: {board.fen()}")
        
        # 短暂暂停以便观察
        await asyncio.sleep(0.3)
    
    return on_move


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='ChessRL 训练脚本')
    
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--agents', type=int, default=8, help='智能体数量')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    parser.add_argument('--gui', action='store_true', help='启用 GUI 可视化窗口')
    parser.add_argument('--ascii', action='store_true', help='启用 ASCII 终端显示')
    parser.add_argument('--viz-interval', type=int, default=10, help='可视化间隔（多少轮显示一次）')
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(run_training(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_agents=args.agents,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose,
            gui_viz=args.gui,
            ascii_viz=args.ascii,
            viz_interval=args.viz_interval
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
