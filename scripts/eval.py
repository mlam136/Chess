#!/usr/bin/env python3
"""
评估脚本 - 模型 vs 模型 / 模型 vs 引擎

用法:
    python scripts/eval.py --model1 path/to/model1.pt --model2 path/to/model2.pt
    python scripts/eval.py --model path/to/model.pt --engine stockfish --depth 10
"""

import argparse
import asyncio
from typing import Optional, Tuple


async def run_evaluation(
    model1_path: Optional[str] = None,
    model2_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    engine_depth: int = 10,
    num_games: int = 20,
    verbose: bool = False
):
    """
    运行评估
    
    Args:
        model1_path: 第一个模型路径
        model2_path: 第二个模型路径（可选）
        engine_path: 引擎路径（如 stockfish）
        engine_depth: 引擎搜索深度
        num_games: 对局数量
        verbose: 详细日志
    """
    print("=" * 60)
    print("ChessRL - 评估模式")
    print("=" * 60)
    
    from model import AlphaZeroResNet, load_checkpoint
    from agent import ModelAgent, create_model_agent
    from engine.game import Game
    from engine.board import Board
    
    # 加载模型
    models = []
    agents = []
    
    if model1_path:
        print(f"加载模型 1: {model1_path}")
        model1 = AlphaZeroResNet(num_res_blocks=8, channels=128)
        load_checkpoint(model1_path, model1)
        
        agent1 = ModelAgent(
            agent_id="model_1",
            model=model1,
            is_teacher=False,
            mcts_iterations=100  # 评估时使用更多迭代
        )
        agents.append(agent1)
        models.append(model1)
    
    if model2_path:
        print(f"加载模型 2: {model2_path}")
        model2 = AlphaZeroResNet(num_res_blocks=8, channels=128)
        load_checkpoint(model2_path, model2)
        
        agent2 = ModelAgent(
            agent_id="model_2",
            model=model2,
            is_teacher=False,
            mcts_iterations=100
        )
        agents.append(agent2)
        models.append(model2)
    
    # 如果指定了引擎
    if engine_path:
        print(f"加载引擎：{engine_path} (depth={engine_depth})")
        try:
            import chess.engine
            transport, engine = await chess.engine.popen_uci(engine_path)
            
            class EngineAgent:
                def __init__(self, agent_id: str, engine, depth: int):
                    self.agent_id = agent_id
                    self.engine = engine
                    self.depth = depth
                
                async def think(self, board: Board) -> str:
                    result = await self.engine.play(
                        board, 
                        chess.engine.Limit(depth=self.depth)
                    )
                    return result.move.uci()
                
                def reset(self):
                    pass
                
                def on_game_end(self, result: str, opponent_id: str):
                    pass
            
            engine_agent = EngineAgent("stockfish", engine, engine_depth)
            agents.append(engine_agent)
            
        except ImportError:
            print("错误：需要安装 python-chess 来使用引擎功能")
            return
        except Exception as e:
            print(f"警告：无法加载引擎 {engine_path}: {e}")
    
    if len(agents) < 2:
        print("错误：至少需要两个对手（模型或引擎）")
        return
    
    # 运行评估对局
    print(f"\n开始评估 - {num_games} 局")
    print(f"对阵：{agents[0].agent_id} vs {agents[1].agent_id}")
    print("=" * 60)
    
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'games': []
    }
    
    for game_num in range(1, num_games + 1):
        # 交替颜色
        if game_num % 2 == 1:
            white_agent, black_agent = agents[0], agents[1]
        else:
            white_agent, black_agent = agents[1], agents[0]
        
        # 创建游戏
        game = Game(
            game_id=f"eval_{game_num}",
            white_player=white_agent,
            black_player=black_agent,
            max_moves=500
        )
        
        # 执行游戏
        if verbose:
            print(f"\n局 {game_num}: ", end="", flush=True)
        
        result = await game.play()
        
        # 记录结果
        if result['result'] == 'win':
            winner = result.get('winner')
            if winner == white_agent.agent_id or winner == black_agent.agent_id:
                if (game_num % 2 == 1 and winner == agents[0].agent_id) or \
                   (game_num % 2 == 0 and winner == agents[1].agent_id):
                    results['wins'] += 1
                else:
                    results['losses'] += 1
        elif result['result'] == 'draw':
            results['draws'] += 1
        
        results['games'].append({
            'game_num': game_num,
            'white': white_agent.agent_id,
            'black': black_agent.agent_id,
            'result': result['result'],
            'moves': result.get('move_count', 0)
        })
        
        if verbose:
            print(f"{result['result']:4s} ({result.get('move_count', 0):3d} 步)")
        
        # 重置智能体状态
        white_agent.reset()
        black_agent.reset()
    
    # 关闭引擎
    if engine_path and hasattr(engine, 'quit'):
        await engine.quit()
        await transport.close()
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果:")
    print(f"  总对局：{num_games}")
    print(f"  胜：{results['wins']} ({results['wins']/num_games*100:.1f}%)")
    print(f"  负：{results['losses']} ({results['losses']/num_games*100:.1f}%)")
    print(f"  和：{results['draws']} ({results['draws']/num_games*100:.1f}%)")
    
    # Elo 估算
    if results['wins'] + results['losses'] > 0:
        elo_diff = 400 * (results['wins'] / (results['wins'] + results['losses']) - 0.5)
        print(f"\n  估算 Elo 差值：{elo_diff:+.1f}")
    
    return results


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='ChessRL 评估脚本')
    
    parser.add_argument('--model1', type=str, required=True, help='第一个模型路径')
    parser.add_argument('--model2', type=str, default=None, help='第二个模型路径')
    parser.add_argument('--engine', type=str, default=None, help='引擎路径（如 stockfish）')
    parser.add_argument('--depth', type=int, default=10, help='引擎搜索深度')
    parser.add_argument('--games', type=int, default=20, help='对局数量')
    parser.add_argument('--verbose', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(run_evaluation(
            model1_path=args.model1,
            model2_path=args.model2,
            engine_path=args.engine,
            engine_depth=args.depth,
            num_games=args.games,
            verbose=args.verbose
        ))
        
        print("\n评估完成!")
        
    except KeyboardInterrupt:
        print("\n评估被中断")
    except Exception as e:
        print(f"\n错误：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
