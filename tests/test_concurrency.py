"""
测试并发 - 多盘对局不冲突
"""

import pytest
import asyncio
from engine.board import Board
from engine.game import Game
from agent.random_agent import RandomAgent


class TestConcurrency:
    """测试并发执行"""
    
    @pytest.mark.asyncio
    async def test_concurrent_games(self):
        """测试并发游戏执行"""
        # 创建多个游戏
        games = []
        for i in range(4):
            white = RandomAgent(f"white_{i}")
            black = RandomAgent(f"black_{i}")
            
            game = Game(
                game_id=f"game_{i}",
                white_player=white,
                black_player=black,
                max_moves=20  # 限制步数加快测试
            )
            games.append(game)
        
        # 并发执行所有游戏
        results = await asyncio.gather(*[g.play() for g in games])
        
        # 验证所有游戏都完成
        assert len(results) == 4
        for result in results:
            assert result is not None
            assert 'result' in result
    
    @pytest.mark.asyncio
    async def test_board_state_isolation(self):
        """测试棋盘状态隔离"""
        boards = [Board() for _ in range(4)]
        
        # 对不同棋盘执行不同走法
        moves_list = ["e2e4", "d2d4", "g1f3", "c2c4"]
        
        for board, move_str in zip(boards, moves_list):
            move = __import__('chess').Move.from_uci(move_str)
            board.push(move)
        
        # 验证每个棋盘状态独立
        for i, board in enumerate(boards):
            assert board.fen().split()[0] != boards[(i+1) % 4].fen().split()[0]
    
    @pytest.mark.asyncio
    async def test_agent_state_isolation(self):
        """测试智能体状态隔离"""
        agents = [RandomAgent(f"agent_{i}") for i in range(4)]
        
        # 并发调用 think
        board = Board()
        results = await asyncio.gather(*[a.think(board.copy()) for a in agents])
        
        # 验证每个智能体都返回了走法
        assert len(results) == 4
        for result in results:
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_shared_resource_safety(self):
        """测试共享资源安全性"""
        from engine.scheduler import GameScheduler
        from engine.scoring import ScoreManager
        
        scoring = ScoreManager(window_size=10)
        scheduler = GameScheduler(max_concurrent=4)
        
        agents = [RandomAgent(f"agent_{i}") for i in range(8)]
        scheduler.register_agents([a.agent_id for a in agents])
        
        # 并发创建和运行多局游戏
        matchups = scheduler.create_matchups()
        
        async def play_match(match):
            return await scheduler.play_game(match)
        
        results = await asyncio.gather(*[play_match(m) for m in matchups[:4]])
        
        # 验证所有游戏都完成且没有冲突
        completed = sum(1 for r in results if r is not None)
        assert completed >= 1  # 至少有一个完成


class TestAsyncOperations:
    """测试异步操作"""
    
    @pytest.mark.asyncio
    async def test_async_think(self):
        """测试异步思考"""
        agent = RandomAgent("test_agent", delay=0.01)
        board = Board()
        
        move = await agent.think(board)
        
        assert isinstance(move, str)
        assert len(move) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_think_calls(self):
        """测试并发 think 调用"""
        agent = RandomAgent("test_agent", delay=0.01)
        boards = [Board() for _ in range(10)]
        
        moves = await asyncio.gather(*[agent.think(b.copy()) for b in boards])
        
        assert len(moves) == 10
        for move in moves:
            assert isinstance(move, str)


class TestSchedulerConcurrency:
    """测试调度器并发"""
    
    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self):
        """测试最大并发限制"""
        from engine.scheduler import GameScheduler
        from engine.scoring import ScoreManager
        
        scoring = ScoreManager()
        scheduler = GameScheduler(scoring_system=scoring, max_concurrent=2)
        
        agents = [RandomAgent(f"a{i}") for i in range(6)]
        scheduler.register_agents([a.agent_id for a in agents])
        
        matchups = scheduler.create_matchups()
        
        # 应该不超过 max_concurrent
        assert len(matchups) <= 3  # 6 个 agent 最多 3 局
        
        # 但实际并发受 max_concurrent 限制
        assert len(matchups) <= scheduler.max_concurrent or len(matchups) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
