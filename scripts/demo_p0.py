"""P0 阶段演示脚本 - 两个随机智能体在 Pygame 窗口中完成一局"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from engine.board import Board
from engine.game import Game, Player, GameResult
from engine.rules import MoveValidator
from engine.scoring import ScoringSystem
from engine.scheduler import MatchScheduler, MatchConfig
from agent.random_agent import RandomAgent
from viz.app import VisualizationApp


async def random_move_func(board: Board) -> str:
    """随机走法函数（用于演示）"""
    legal_moves = board.legal_moves
    if not legal_moves:
        return ""
    
    import random
    move = random.choice(legal_moves)
    return move.uci()


async def run_p0_demo():
    """
    P0 阶段演示
    两个随机智能体在 Pygame 窗口中对弈
    """
    print("Starting P0 Demo - Random Agents Chess Game")
    print("=" * 50)
    
    # 创建计分系统
    scoring = ScoringSystem(window_size=10)
    
    # 创建调度器
    config = MatchConfig(max_concurrent_games=1)
    scheduler = MatchScheduler(scoring_system=scoring, config=config)
    
    # 注册两个随机智能体
    agent_white = RandomAgent(agent_id="agent_white", delay=0.3)
    agent_black = RandomAgent(agent_id="agent_black", delay=0.3)
    
    scheduler.register_agents([agent_white.agent_id, agent_black.agent_id])
    
    # 创建游戏
    white_player = Player(agent_id=agent_white.agent_id, color=True)
    black_player = Player(agent_id=agent_black.agent_id, color=False)
    
    game = Game(
        game_id="demo_game_001",
        white_player=white_player,
        black_player=black_player,
    )
    
    # 开始游戏
    game.start()
    print(f"Game started: {agent_white.agent_id} (White) vs {agent_black.agent_id} (Black)")
    
    # 创建可视化应用
    app = VisualizationApp(
        width=1000,
        height=600,
        grid_rows=1,
        grid_cols=1,
        scheduler=scheduler,
    )
    
    # 注册游戏到可视化
    app.register_game(game)
    
    print("Pygame window opened. Close it or press ESC to exit.")
    print("Press 'P' to pause/resume.")
    
    # 运行游戏直到结束或用户退出
    max_moves = 200  # 防止无限循环
    
    async def game_loop():
        nonlocal app
        
        move_count = 0
        
        while not game.board.is_game_over and move_count < max_moves:
            # 获取当前玩家
            current_player = game.current_player
            
            # 获取走法
            board_copy = game.board.copy()
            
            if current_player.agent_id == agent_white.agent_id:
                move = await random_move_func(board_copy)
            else:
                move = await random_move_func(board_copy)
            
            if not move:
                print("No legal moves available")
                break
            
            # 执行走法
            result, success = await game.make_move(current_player.agent_id, move)
            
            if success:
                move_count += 1
                san = game.board.internal_board.san(result.move)
                print(f"Move {move_count}: {current_player.agent_id} played {san}")
            
            # 短暂延迟让 UI 更新
            await asyncio.sleep(0.1)
        
        # 游戏结束
        if game.board.is_game_over:
            result_code = game.board.internal_board.result()
            print(f"\nGame ended! Result: {result_code}")
            print(f"Total moves: {move_count}")
            
            # 记录分数
            scoring.record_game(
                game_id=game.game_id,
                white_agent=agent_white.agent_id,
                black_agent=agent_black.agent_id,
                result_code=result_code,
            )
            
            # 打印排名
            rankings = scoring.get_rankings()
            print("\nFinal Rankings:")
            for rank, (agent_id, score) in enumerate(rankings, 1):
                print(f"  {rank}. {agent_id}: {score:.2f}")
        
        # 保持窗口打开几秒钟
        for _ in range(10):
            await asyncio.sleep(0.1)
        
        app.quit()
    
    # 同时运行游戏和 UI
    await asyncio.gather(
        game_loop(),
        app.run_async(),
    )
    
    print("\nP0 Demo completed!")


if __name__ == "__main__":
    try:
        asyncio.run(run_p0_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
