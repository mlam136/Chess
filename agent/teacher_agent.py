"""
Teacher Agent - 冻结参数的教师智能体

继承自 ModelAgent，专门用于知识蒸馏的教师角色
- 参数完全冻结（requires_grad=False）
- 仅执行前向推理，不进行 MCTS 搜索（可选）
- 作为 Student 的学习目标
"""

import asyncio
from typing import Tuple, Optional
import numpy as np
import torch

from engine.board import Board
from .base import AgentType
from .model_agent import ModelAgent
from model.network import AlphaZeroResNet
from model.encoder import encode_board


class TeacherAgent(ModelAgent):
    """
    教师智能体
    
    特点：
    1. 模型参数完全冻结
    2. 可选择是否使用 MCTS（默认直接使用网络输出）
    3. 为 Student 提供软目标（soft targets）
    """
    
    def __init__(self, agent_id: str, model: AlphaZeroResNet,
                 use_mcts: bool = False, mcts_iterations: int = 20,
                 temperature: float = 1.0):
        """
        初始化教师智能体
        
        Args:
            agent_id: 智能体 ID
            model: 神经网络模型（应已冻结）
            use_mcts: 是否使用 MCTS 搜索（默认 False，直接网络推理）
            mcts_iterations: MCTS 迭代次数（如果使用 MCTS）
            temperature: 策略温度
        """
        super().__init__(
            agent_id=agent_id,
            model=model,
            is_teacher=True,  # 强制设为 Teacher
            mcts_iterations=mcts_iterations,
            temperature=temperature
        )
        
        self.use_mcts = use_mcts
    
    async def think(self, board: Board) -> str:
        """
        思考并返回走法
        
        Teacher 可以选择：
        1. 直接使用网络策略（快速）
        2. 使用 MCTS 搜索（更精确，但慢）
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            str: UCI 格式走法
        """
        if self.use_mcts:
            # 使用 MCTS 搜索（更精确的走法）
            pi_mcts, value = await self._mcts_search_async(board)
            move = self._select_move_from_policy(pi_mcts, board)
        else:
            # 直接使用网络策略（快速推理）
            pi_net, value = self.get_policy_and_value(board)
            move = self._select_move_from_policy(pi_net, board)
        
        if move:
            return move.uci()
        else:
            return ""
    
    def get_teacher_output(self, board: Board) -> Tuple[np.ndarray, float]:
        """
        获取教师输出（用于蒸馏）
        
        返回软目标供 Student 学习
        
        Args:
            board: 棋盘状态
            
        Returns:
            policy: [4608] 策略概率分布（软目标）
            value: 标量价值估计
        """
        return self.get_policy_and_value(board)
    
    def set_model(self, model: AlphaZeroResNet) -> None:
        """
        更新模型（当 Teacher 权重需要更新时调用）
        
        注意：调用后会自动冻结新模型
        """
        self.model = model
        self._freeze_model()
        self.device = next(model.parameters()).device


def create_teacher_agent(agent_id: str, model: AlphaZeroResNet,
                         use_mcts: bool = False, **kwargs) -> TeacherAgent:
    """
    创建教师智能体的工厂函数
    
    Args:
        agent_id: 智能体 ID
        model: 神经网络模型
        use_mcts: 是否使用 MCTS
        **kwargs: 其他参数
        
    Returns:
        TeacherAgent 实例
    """
    return TeacherAgent(
        agent_id=agent_id,
        model=model,
        use_mcts=use_mcts,
        **kwargs
    )
