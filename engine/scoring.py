"""
滑动窗口计分、身份分配
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class GameRecord:
    """单局游戏记录"""
    game_id: str
    agent_id: str
    opponent_id: str
    color: bool  # True=白方，False=黑方
    result: str  # "win", "loss", "draw"
    score: float  # 1.0=胜，0.5=和，0.0=负


@dataclass
class AgentScore:
    """智能体分数信息"""
    agent_id: str
    total_score: float = 0.0      # 总分
    games_played: int = 0         # 总局数
    wins: int = 0                 # 胜场
    losses: int = 0               # 负场
    draws: int = 0                # 和棋
    win_rate: float = 0.0         # 胜率
    
    def update(self, result: str) -> None:
        """更新统计数据"""
        self.games_played += 1
        
        if result == "win":
            self.wins += 1
            self.total_score += 1.0
        elif result == "draw":
            self.draws += 1
            self.total_score += 0.5
        elif result == "loss":
            self.losses += 1
            self.total_score += 0.0
        
        if self.games_played > 0:
            self.win_rate = self.wins / self.games_played
    
    @property
    def average_score(self) -> float:
        """平均分"""
        if self.games_played == 0:
            return 0.0
        return self.total_score / self.games_played


class ScoringSystem:
    """
    计分系统
    使用滑动窗口计算最近 N 局的平均表现
    """
    
    def __init__(self, window_size: int = 10):
        """
        初始化计分系统
        
        Args:
            window_size: 滑动窗口大小（最近 N 局）
        """
        self.window_size = window_size
        self._agent_scores: Dict[str, AgentScore] = {}
        self._recent_games: Dict[str, deque] = {}  # 每个 agent 的最近游戏记录
    
    def register_agent(self, agent_id: str) -> None:
        """注册智能体"""
        if agent_id not in self._agent_scores:
            self._agent_scores[agent_id] = AgentScore(agent_id=agent_id)
            self._recent_games[agent_id] = deque(maxlen=self.window_size)
    
    def record_game(
        self,
        game_id: str,
        white_agent: str,
        black_agent: str,
        result_code: str  # "1-0", "0-1", "1/2-1/2"
    ) -> None:
        """
        记录一局游戏结果
        
        Args:
            game_id: 游戏 ID
            white_agent: 白方智能体 ID
            black_agent: 黑方智能体 ID
            result_code: 游戏结果代码
        """
        # 确保智能体已注册
        self.register_agent(white_agent)
        self.register_agent(black_agent)
        
        # 解析结果
        if result_code == "1-0":
            white_result = "win"
            black_result = "loss"
        elif result_code == "0-1":
            white_result = "loss"
            black_result = "win"
        else:  # "1/2-1/2"
            white_result = "draw"
            black_result = "draw"
        
        # 创建记录
        white_record = GameRecord(
            game_id=game_id,
            agent_id=white_agent,
            opponent_id=black_agent,
            color=True,
            result=white_result,
            score=1.0 if white_result == "win" else (0.5 if white_result == "draw" else 0.0)
        )
        
        black_record = GameRecord(
            game_id=game_id,
            agent_id=black_agent,
            opponent_id=white_agent,
            color=False,
            result=black_result,
            score=1.0 if black_result == "win" else (0.5 if black_result == "draw" else 0.0)
        )
        
        # 更新统计
        self._agent_scores[white_agent].update(white_result)
        self._agent_scores[black_agent].update(black_result)
        
        # 添加到滑动窗口
        self._recent_games[white_agent].append(white_record)
        self._recent_games[black_agent].append(black_record)
    
    def get_score(self, agent_id: str) -> Optional[AgentScore]:
        """获取智能体总分统计"""
        return self._agent_scores.get(agent_id)
    
    def get_window_score(self, agent_id: str) -> float:
        """
        获取滑动窗口内的平均分
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            float: 窗口内平均分
        """
        if agent_id not in self._recent_games:
            return 0.0
        
        recent = self._recent_games[agent_id]
        if len(recent) == 0:
            return 0.0
        
        total = sum(record.score for record in recent)
        return total / len(recent)
    
    def get_all_scores(self) -> Dict[str, AgentScore]:
        """获取所有智能体的分数"""
        return self._agent_scores.copy()
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        获取排名（按滑动窗口分数降序）
        
        Returns:
            List[Tuple[agent_id, window_score]]: 排名列表
        """
        rankings = [
            (agent_id, self.get_window_score(agent_id))
            for agent_id in self._agent_scores.keys()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_top_agents(self, n: int) -> List[str]:
        """获取前 N 名智能体 ID"""
        rankings = self.get_rankings()
        return [agent_id for agent_id, _ in rankings[:n]]
    
    def get_bottom_agents(self, n: int) -> List[str]:
        """获取后 N 名智能体 ID"""
        rankings = self.get_rankings()
        return [agent_id for agent_id, _ in rankings[-n:]]
    
    def reset(self) -> None:
        """重置所有分数"""
        self._agent_scores.clear()
        self._recent_games.clear()
    
    def set_window_size(self, size: int) -> None:
        """设置滑动窗口大小"""
        self.window_size = size
        # 重新调整所有 deque 的大小
        for agent_id in self._recent_games:
            old_deque = self._recent_games[agent_id]
            new_deque = deque(old_deque, maxlen=size)
            self._recent_games[agent_id] = new_deque
