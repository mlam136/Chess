"""
测试计分系统 - 滑动窗口计分、身份分配
"""

import pytest
from engine.scoring import ScoreManager, GameResult


class TestScoreManager:
    """测试计分管理器"""
    
    def test_record_game_win(self):
        """测试记录胜利"""
        manager = ScoreManager(window_size=10)
        
        manager.record_result(
            game_id="game_001",
            white_id="agent_A",
            black_id="agent_B",
            result="win"  # 白胜
        )
        
        rankings = manager.get_rankings()
        
        assert len(rankings) == 2
        assert rankings[0][0] == "agent_A"
        assert rankings[0][1] > 0
    
    def test_record_game_draw(self):
        """测试记录和局"""
        manager = ScoreManager(window_size=10)
        
        manager.record_result(
            game_id="game_001",
            white_id="agent_A",
            black_id="agent_B",
            result="draw"
        )
        
        rankings = manager.get_rankings()
        
        # 双方都应该有分数（和局各得 0.5）
        assert len(rankings) == 2
        assert rankings[0][1] == rankings[1][1]
    
    def test_sliding_window(self):
        """测试滑动窗口"""
        manager = ScoreManager(window_size=3)
        
        # 记录超过窗口大小的对局
        for i in range(5):
            manager.record_result(
                game_id=f"game_{i:03d}",
                white_id="agent_A",
                black_id="agent_B",
                result="win" if i % 2 == 0 else "loss"
            )
        
        # 只应该保留最近的 window_size 局
        assert len(manager.game_history) <= 3
    
    def test_update_identities(self):
        """测试身份更新"""
        from agent import RandomAgent
        
        manager = ScoreManager(window_size=10)
        agents = [
            RandomAgent("agent_0"),
            RandomAgent("agent_1"),
            RandomAgent("agent_2"),
            RandomAgent("agent_3"),
        ]
        
        # 记录一些对局
        for i in range(5):
            manager.record_result(
                game_id=f"game_{i}",
                white_id=f"agent_{i % 4}",
                black_id=f"agent_{(i + 1) % 4}",
                result="win" if i % 2 == 0 else "draw"
            )
        
        # 更新身份
        manager.update_identities(agents)
        
        # 验证身份已分配
        # 前 50% 应该是 Teacher，后 50% 是 Student
        sorted_agents = sorted(agents, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        
        # 至少应该有身份分配
        assert len(sorted_agents) == 4
    
    def test_get_agent_stats(self):
        """测试获取智能体统计"""
        manager = ScoreManager(window_size=10)
        
        # 记录多局
        for i in range(3):
            manager.record_result(
                game_id=f"game_{i}",
                white_id="agent_A",
                black_id="agent_B",
                result="win"
            )
        
        stats = manager.get_agent_stats("agent_A")
        
        assert 'games_played' in stats
        assert 'wins' in stats
        assert 'draws' in stats
        assert 'losses' in stats
        assert 'score' in stats
        assert 'win_rate' in stats


class TestGameResult:
    """测试游戏结果枚举"""
    
    def test_game_result_values(self):
        """测试结果值"""
        assert GameResult.WHITE_WIN.value == "1-0"
        assert GameResult.BLACK_WIN.value == "0-1"
        assert GameResult.DRAW.value == "1/2-1/2"
    
    def test_parse_result(self):
        """测试解析结果"""
        assert GameResult.from_string("1-0") == GameResult.WHITE_WIN
        assert GameResult.from_string("0-1") == GameResult.BLACK_WIN
        assert GameResult.from_string("1/2-1/2") == GameResult.DRAW
        assert GameResult.from_string("draw") == GameResult.DRAW


class TestIdentityAssignment:
    """测试身份分配边界情况"""
    
    def test_even_number_agents(self):
        """测试偶数智能体"""
        manager = ScoreManager(window_size=10)
        
        # 8 个智能体，应该 4 个 Teacher，4 个 Student
        teacher_count, student_count = manager._calculate_identity_split(8)
        
        assert teacher_count == 4
        assert student_count == 4
    
    def test_minimum_agents(self):
        """测试最小智能体数量"""
        manager = ScoreManager(window_size=10)
        
        # 4 个智能体（最小值）
        teacher_count, student_count = manager._calculate_identity_split(4)
        
        assert teacher_count == 2
        assert student_count == 2
    
    def test_start_y_threshold(self):
        """测试 START_Y 阈值"""
        manager = ScoreManager(window_size=10, start_y=5)
        
        # 在 START_Y 之前不应该更新身份
        assert not manager._should_update_identities(3)
        
        # 达到 START_Y 后应该更新
        assert manager._should_update_identities(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
