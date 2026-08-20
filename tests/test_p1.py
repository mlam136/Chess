"""
P1 阶段测试 - 并发 & 计分
测试 scheduler.py + scoring.py + api/ 的功能
"""

import pytest
import asyncio
from engine.scheduler import MatchScheduler, MatchConfig
from engine.scoring import ScoringSystem
from api.events import EventBus, EventType, get_event_bus
from api.board_state import BoardStateManager
from engine.board import Board


class TestScoringSystem:
    """测试滑动窗口计分系统"""
    
    def test_register_agent(self):
        """测试注册智能体"""
        scoring = ScoringSystem(window_size=10)
        scoring.register_agent('agent_A')
        
        score = scoring.get_score('agent_A')
        assert score is not None
        assert score.agent_id == 'agent_A'
        assert score.games_played == 0
    
    def test_record_game_win_loss(self):
        """测试记录胜负"""
        scoring = ScoringSystem(window_size=10)
        scoring.register_agent('A')
        scoring.register_agent('B')
        
        # A 胜 B
        scoring.record_game('game_001', 'A', 'B', '1-0')
        
        score_a = scoring.get_score('A')
        score_b = scoring.get_score('B')
        
        assert score_a.wins == 1
        assert score_a.total_score == 1.0
        assert score_b.losses == 1
        assert score_b.total_score == 0.0
    
    def test_record_game_draw(self):
        """测试记录和棋"""
        scoring = ScoringSystem(window_size=10)
        scoring.register_agent('A')
        scoring.register_agent('B')
        
        scoring.record_game('game_001', 'A', 'B', '1/2-1/2')
        
        score_a = scoring.get_score('A')
        score_b = scoring.get_score('B')
        
        assert score_a.draws == 1
        assert score_a.total_score == 0.5
        assert score_b.draws == 1
        assert score_b.total_score == 0.5
    
    def test_window_score(self):
        """测试滑动窗口平均分"""
        scoring = ScoringSystem(window_size=3)
        scoring.register_agent('A')
        
        # 记录 5 局，但只计算最近 3 局
        scoring.record_game('g1', 'A', 'B', '1-0')  # 1.0
        scoring.record_game('g2', 'A', 'B', '0-1')  # 0.0
        scoring.record_game('g3', 'A', 'B', '1-0')  # 1.0
        scoring.record_game('g4', 'A', 'B', '0-1')  # 0.0 (超出窗口)
        scoring.record_game('g5', 'A', 'B', '1-0')  # 1.0
        
        # 窗口内应该是 g3, g4, g5: (1.0 + 0.0 + 1.0) / 3 = 0.667
        window_score = scoring.get_window_score('A')
        assert abs(window_score - 0.667) < 0.01
    
    def test_rankings(self):
        """测试排名"""
        scoring = ScoringSystem(window_size=10)
        
        for i in range(4):
            scoring.register_agent(f'agent_{i}')
        
        # 制造不同分数
        scoring.record_game('g1', 'agent_0', 'agent_1', '1-0')
        scoring.record_game('g2', 'agent_2', 'agent_3', '1-0')
        scoring.record_game('g3', 'agent_0', 'agent_2', '1-0')
        
        rankings = scoring.get_rankings()
        
        # agent_0 应该排第一（2 胜）
        assert rankings[0][0] == 'agent_0'
        assert rankings[0][1] == 1.0  # 2 胜 / 2 局


class TestMatchScheduler:
    """测试匹配调度器"""
    
    @pytest.mark.asyncio
    async def test_register_agents(self):
        """测试注册智能体"""
        scoring = ScoringSystem()
        scheduler = MatchScheduler(scoring_system=scoring)
        
        agent_ids = [f'agent_{i}' for i in range(8)]
        scheduler.register_agents(agent_ids)
        
        # 检查所有智能体都已注册到计分系统
        for agent_id in agent_ids:
            assert scoring.get_score(agent_id) is not None
    
    @pytest.mark.asyncio
    async def test_random_pairing(self):
        """测试随机配对"""
        scoring = ScoringSystem()
        config = MatchConfig(max_concurrent_games=4)
        scheduler = MatchScheduler(scoring_system=scoring, config=config)
        
        agent_ids = [f'agent_{i}' for i in range(8)]
        scheduler.register_agents(agent_ids)
        
        games = await scheduler.start_match_round(use_ranking=False)
        
        # 应该创建 4 个游戏（最大并发数）
        assert len(games) == 4
        
        # 检查所有游戏都在进行中
        active_games = scheduler.get_active_games()
        assert len(active_games) == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """测试并发限制"""
        scoring = ScoringSystem()
        config = MatchConfig(max_concurrent_games=2)
        scheduler = MatchScheduler(scoring_system=scoring, config=config)
        
        agent_ids = [f'agent_{i}' for i in range(8)]
        scheduler.register_agents(agent_ids)
        
        games = await scheduler.start_match_round(use_ranking=False)
        
        # 应该只创建 2 个游戏（受限于 max_concurrent_games）
        assert len(games) <= 2
    
    @pytest.mark.asyncio
    async def test_game_completion(self):
        """测试游戏完成处理"""
        scoring = ScoringSystem()
        config = MatchConfig(max_concurrent_games=4)
        scheduler = MatchScheduler(scoring_system=scoring, config=config)
        
        agent_ids = ['agent_A', 'agent_B']
        scheduler.register_agents(agent_ids)
        
        games = await scheduler.start_match_round(use_ranking=False)
        
        # 模拟游戏结束 - 每个游戏使用不同的结果
        from engine.game import GameResult
        
        for i, game in enumerate(games):
            # 根据实际玩家分配胜负
            # game.white_player.agent_id 和 game.black_player.agent_id 是实际的玩家
            # result_code='1-0' 表示白方胜，'0-1' 表示黑方胜
            if i % 2 == 0:
                result_code = '1-0'  # 白方胜
                winner = game.white_player.agent_id
            else:
                result_code = '0-1'  # 黑方胜
                winner = game.black_player.agent_id
            
            result = GameResult(
                winner=winner,
                result_code=result_code,
                reason='checkmate',
                move_count=10,
                pgn=''
            )
            
            scheduler._on_game_end(game.game_id, result)
        
        # 检查游戏已移动到已完成列表
        assert len(scheduler.get_active_games()) == 0
        assert len(scheduler.get_completed_games()) == len(games)
        
        # 检查分数已记录 - 两个智能体都应该有游戏记录
        score_a = scoring.get_score('agent_A')
        score_b = scoring.get_score('agent_B')
        
        # 总共应该有 len(games) 局游戏记录
        assert score_a.games_played + score_b.games_played == len(games) * 2


class TestEventBus:
    """测试事件总线"""
    
    def test_subscribe_publish(self):
        """测试订阅和发布"""
        bus = EventBus()
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        bus.subscribe(EventType.GAME_STARTED, handler)
        bus.publish(EventType.GAME_STARTED, {'game_id': 'test'})
        
        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.GAME_STARTED
    
    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = EventBus()
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        bus.subscribe(EventType.GAME_ENDED, handler)
        bus.unsubscribe(EventType.GAME_ENDED, handler)
        bus.publish(EventType.GAME_ENDED, {})
        
        assert len(events_received) == 0
    
    def test_event_history(self):
        """测试事件历史"""
        # 创建新的事件总线实例，避免与其他测试共享状态
        bus = EventBus()
        bus.clear_history()  # 清空之前的历史
        
        events_received = []
        def handler(event):
            events_received.append(event)
        
        bus.subscribe(EventType.GAME_STARTED, handler)
        bus.subscribe(EventType.GAME_ENDED, handler)
        bus.subscribe(EventType.SCORE_UPDATED, handler)
        
        bus.publish(EventType.GAME_STARTED, {'id': 1})
        bus.publish(EventType.GAME_ENDED, {'id': 2})
        bus.publish(EventType.SCORE_UPDATED, {'id': 3})
        
        history = bus.get_history(limit=10)
        assert len(history) == 3
        
        # 按类型过滤
        started_events = bus.get_history(EventType.GAME_STARTED)
        assert len(started_events) == 1


class TestBoardStateManager:
    """测试棋盘状态管理"""
    
    def test_register_and_get_state(self):
        """测试注册和获取状态"""
        manager = BoardStateManager()
        board = Board()
        
        manager.register_board('game_001', board)
        
        state = manager.get_state('game_001')
        assert state is not None
        assert state.fen == board.fen
    
    def test_update_state(self):
        """测试更新状态"""
        manager = BoardStateManager()
        board = Board()
        
        manager.register_board('game_001', board)
        
        # 走一步
        board.make_move_uci('e2e4')
        
        # 更新状态
        manager.update_state('game_001')
        
        state = manager.get_state('game_001')
        assert state.fen != Board().fen  # FEN 应该改变
    
    def test_get_legal_moves(self):
        """测试获取合法走法"""
        manager = BoardStateManager()
        board = Board()
        
        manager.register_board('game_001', board)
        
        moves = manager.get_legal_moves('game_001')
        assert len(moves) == 20  # 初始局面有 20 种合法走法
    
    def test_multiple_boards(self):
        """测试多棋盘管理"""
        manager = BoardStateManager()
        
        for i in range(4):
            board = Board()
            manager.register_board(f'game_{i:03d}', board)
        
        all_states = manager.get_all_states()
        assert len(all_states) == 4


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_match_round(self):
        """测试完整的一轮匹配"""
        # 创建所有组件
        scoring = ScoringSystem(window_size=10)
        config = MatchConfig(max_concurrent_games=4)
        scheduler = MatchScheduler(scoring_system=scoring, config=config)
        board_manager = BoardStateManager()
        
        # 注册 8 个智能体
        agent_ids = [f'agent_{i}' for i in range(8)]
        scheduler.register_agents(agent_ids)
        
        # 开始一轮匹配
        games = await scheduler.start_match_round(use_ranking=False)
        assert len(games) == 4
        
        # 为每个游戏创建棋盘
        for game in games:
            board = Board()
            board_manager.register_board(game.game_id, board)
        
        # 模拟游戏结束
        from engine.game import GameResult
        results = ['1-0', '0-1', '1/2-1/2', '1-0']
        
        for i, game in enumerate(games):
            result_code = results[i % len(results)]
            scoring.record_game(
                game_id=game.game_id,
                white_agent=game.white_player.agent_id,
                black_agent=game.black_player.agent_id,
                result_code=result_code
            )
            
            result = GameResult(
                winner=None if result_code == '1/2-1/2' else (
                    game.white_player.agent_id if result_code == '1-0' else game.black_player.agent_id
                ),
                result_code=result_code,
                reason='test',
                move_count=game.board.move_count,
                pgn=''
            )
            scheduler._on_game_end(game.game_id, result)
        
        # 检查排名和身份分配
        rankings = scoring.get_rankings()
        assert len(rankings) == 8
        
        # 前 50% 是 Teacher，后 50% 是 Student
        n_agents = len(rankings)
        half = n_agents // 2
        teacher_ids = [agent_id for agent_id, _ in rankings[:half]]
        student_ids = [agent_id for agent_id, _ in rankings[half:]]
        
        assert len(teacher_ids) == 4
        assert len(student_ids) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
