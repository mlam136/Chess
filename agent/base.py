"""
Agent 抽象基类
"""

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
import chess

from engine.board import Board


class AgentType(Enum):
    """智能体类型"""
    RANDOM = "random"           # 随机走子
    MODEL = "model"             # 模型驱动（MCTS + Neural Network）
    TEACHER = "teacher"         # Teacher（冻结的模型）
    HUMAN = "human"             # 人类玩家


class Agent(ABC):
    """
    智能体抽象基类
    所有棋手必须继承此类
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体唯一标识
            agent_type: 智能体类型
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
    
    @abstractmethod
    async def think(self, board: Board) -> str:
        """
        思考并返回走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            str: UCI 格式的走法（如 "e2e4"）
        """
        pass
    
    def reset(self) -> None:
        """重置智能体状态（每局新游戏时调用）"""
        pass
    
    def on_game_end(self, result: str, opponent_id: str) -> None:
        """
        游戏结束通知
        
        Args:
            result: 游戏结果 ("win", "loss", "draw")
            opponent_id: 对手 ID
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.agent_type.value}({self.agent_id})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} type={self.agent_type.value}>"
