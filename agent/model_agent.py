"""
Model Agent - 基于 MCTS + Neural Network 的智能体
"""

import asyncio
from typing import Optional, Tuple
import numpy as np
import torch
import chess

from engine.board import Board
from .base import Agent, AgentType
from mcts.search import mcts_search
from model.network import AlphaZeroResNet
from model.encoder import encode_board


class ModelAgent(Agent):
    """
    模型驱动智能体
    
    使用 MCTS + Neural Network 进行决策
    支持 Teacher 和 Student 两种模式：
    - Teacher: 冻结参数，仅推理
    - Student: 可训练，使用 MCTS 搜索
    """
    
    def __init__(self, agent_id: str, model: AlphaZeroResNet, 
                 is_teacher: bool = False, mcts_iterations: int = 50,
                 temperature: float = 1.0, think_delay: float = 0.0):
        """
        初始化模型智能体
        
        Args:
            agent_id: 智能体 ID
            model: 神经网络模型
            is_teacher: 是否为 Teacher（冻结参数）
            mcts_iterations: MCTS 迭代次数
            temperature: 策略温度（>1 增加随机性）
            think_delay: 思考延迟（秒）
        """
        super().__init__(agent_id, AgentType.TEACHER if is_teacher else AgentType.MODEL)
        
        self.model = model
        self.is_teacher = is_teacher
        self.mcts_iterations = mcts_iterations
        self.temperature = temperature
        self.think_delay = think_delay
        
        # 如果为 Teacher，冻结参数
        if is_teacher:
            self._freeze_model()
        
        # 设备
        self.device = next(model.parameters()).device
        
        # 统计信息
        self.total_games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
    
    def _freeze_model(self):
        """冻结模型参数（Teacher 模式）"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def _unfreeze_model(self):
        """解冻模型参数（Student 模式）"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
    
    async def think(self, board: Board) -> str:
        """
        思考并返回走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            str: UCI 格式走法
        """
        # 模拟思考延迟
        if self.think_delay > 0:
            await asyncio.sleep(self.think_delay)
        
        # MCTS 搜索
        pi_mcts, value = await self._mcts_search_async(board)
        
        # 从策略分布中选择走法
        move = self._select_move_from_policy(pi_mcts, board)
        
        if move:
            return move.uci()
        else:
            # 无合法走法
            return ""
    
    async def _mcts_search_async(self, board: Board) -> Tuple[np.ndarray, float]:
        """
        异步执行 MCTS 搜索
        
        Args:
            board: 棋盘状态
            
        Returns:
            pi_mcts: [4608] 策略分布
            value: 价值估计
        """
        # 在后台线程中执行 MCTS（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        
        def _search():
            return mcts_search(
                board=board,
                model=self.model if not self.is_teacher else self.model,
                iterations=self.mcts_iterations,
                c_puct=1.5
            )
        
        pi_mcts, value = await loop.run_in_executor(None, _search)
        return pi_mcts, value
    
    def _select_move_from_policy(self, pi: np.ndarray, board: Board) -> Optional[chess.Move]:
        """
        从策略分布中选择走法
        
        Args:
            pi: [4608] 策略分布
            board: 棋盘状态
            
        Returns:
            选中的走法
        """
        import chess
        from mcts.policy import policy_to_move, best_move_from_policy
        
        legal_moves = board.legal_moves
        
        if not legal_moves:
            return None
        
        if self.temperature == 0:
            # 贪婪选择
            return best_move_from_policy(pi, legal_moves)
        else:
            # 带温度采样
            return policy_to_move(pi, legal_moves, temperature=self.temperature)
    
    def get_policy_and_value(self, board: Board) -> Tuple[np.ndarray, float]:
        """
        获取当前局面的策略和价值（用于蒸馏）
        
        Args:
            board: 棋盘状态
            
        Returns:
            policy: [4608] 策略概率
            value: 标量价值
        """
        with torch.no_grad():
            # 编码棋盘
            state_tensor = encode_board(board)
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            
            # 模型推理
            policy_logits, value = self.model(state_tensor)
            
            # 处理输出
            policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            value_scalar = value.squeeze().item()
        
        return policy_probs, value_scalar
    
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
        self.total_games += 1
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        elif result == "loss":
            self.losses += 1
    
    @property
    def win_rate(self) -> float:
        """获取胜率"""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games
    
    @property
    def score(self) -> float:
        """获取平均得分（胜=1, 和=0.5, 负=0）"""
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games
    
    def set_temperature(self, temp: float) -> None:
        """设置策略温度"""
        self.temperature = temp
    
    def set_mcts_iterations(self, iterations: int) -> None:
        """设置 MCTS 迭代次数"""
        self.mcts_iterations = iterations
    
    def __str__(self) -> str:
        role = "Teacher" if self.is_teacher else "Student"
        return f"{role}({self.agent_id})"


def create_model_agent(agent_id: str, is_teacher: bool = False,
                       num_res_blocks: int = 8, channels: int = 128,
                       pretrained_path: str = None, **kwargs) -> ModelAgent:
    """
    创建模型智能体的工厂函数
    
    Args:
        agent_id: 智能体 ID
        is_teacher: 是否为 Teacher
        num_res_blocks: 残差块数量
        channels: 中间层通道数
        pretrained_path: 预训练权重路径
        **kwargs: 其他参数（mcts_iterations, temperature, think_delay）
        
    Returns:
        ModelAgent 实例
    """
    # 创建模型
    model = AlphaZeroResNet(num_res_blocks=num_res_blocks, channels=channels)
    
    # 加载预训练权重
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建智能体
    agent = ModelAgent(
        agent_id=agent_id,
        model=model,
        is_teacher=is_teacher,
        **kwargs
    )
    
    return agent
