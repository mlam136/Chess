"""
随机走子智能体（用于 P0 阶段测试）
"""

import random
import asyncio
from typing import List

from engine.board import Board
from .base import Agent, AgentType


class RandomAgent(Agent):
    """
    随机智能体
    从合法走法中随机选择一个
    
    用于 P0 阶段的引擎和可视化测试
    """
    
    def __init__(self, agent_id: str, delay: float = 0.1):
        """
        初始化随机智能体
        
        Args:
            agent_id: 智能体 ID
            delay: 思考延迟（秒），模拟真实思考时间
        """
        super().__init__(agent_id, AgentType.RANDOM)
        self.delay = delay
    
    async def think(self, board: Board) -> str:
        """
        随机选择一个合法走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            str: UCI 格式走法
        """
        # 模拟思考延迟
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        legal_moves = board.legal_moves
        
        if not legal_moves:
            # 无合法走法，返回空字符串（游戏应该已经结束）
            return ""
        
        # 随机选择
        move = random.choice(legal_moves)
        
        # 返回 UCI 格式
        return move.uci()
    
    def reset(self) -> None:
        """重置状态（随机智能体无需特殊处理）"""
        pass
    
    def on_game_end(self, result: str, opponent_id: str) -> None:
        """游戏结束通知（随机智能体无需特殊处理）"""
        pass


def create_random_agents(n: int, delay: float = 0.1) -> List[RandomAgent]:
    """
    批量创建随机智能体
    
    Args:
        n: 智能体数量
        delay: 思考延迟
        
    Returns:
        List[RandomAgent]: 智能体列表
    """
    agents = []
    for i in range(n):
        agent = RandomAgent(agent_id=f"random_{i:02d}", delay=delay)
        agents.append(agent)
    
    return agents
