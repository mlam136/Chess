"""
MCTS 节点定义
"""

import numpy as np
from typing import Optional, Dict, List
import chess

from engine.board import Board


class MCTSNode:
    """
    蒙特卡洛树搜索节点
    
    Attributes:
        board: 当前棋盘状态
        parent: 父节点
        children: 子节点字典 {move: MCTSNode}
        Q: 动作价值估计 (期望回报)
        N: 访问次数
        P: 先验概率 (由神经网络输出)
        is_terminal: 是否为终止节点
    """
    
    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None, 
                 prior: float = 0.0):
        """
        初始化 MCTS 节点
        
        Args:
            board: 当前棋盘状态
            parent: 父节点
            prior: 先验概率 P(s,a)
        """
        self.board = board
        self.parent = parent
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # PUCT 算法核心变量
        self.Q = 0.0  # 动作价值
        self.N = 0    # 访问次数
        self.P = prior  # 先验概率
        
        # 节点状态
        self.is_terminal = board.is_game_over
        
        # 合法走法（延迟计算）
        self._legal_moves: Optional[List[chess.Move]] = None
    
    @property
    def legal_moves(self) -> List[chess.Move]:
        """获取合法走法列表（缓存）"""
        if self._legal_moves is None:
            self._legal_moves = self.board.legal_moves
        return self._legal_moves
    
    @property
    def is_fully_expanded(self) -> bool:
        """检查是否已完全扩展（所有合法走法都已创建子节点）"""
        if self.is_terminal:
            return True
        return len(self.children) >= len(self.legal_moves)
    
    @property
    def value(self) -> float:
        """
        返回节点价值估计
        对于叶子节点，这是神经网络的评估值
        对于内部节点，这是子节点的加权平均
        """
        if self.N == 0:
            return 0.0
        return self.Q
    
    def get_child(self, move: chess.Move) -> Optional['MCTSNode']:
        """获取指定走法的子节点"""
        return self.children.get(move)
    
    def add_child(self, move: chess.Move, prior: float) -> 'MCTSNode':
        """
        添加子节点
        
        Args:
            move: 走法
            prior: 先验概率
            
        Returns:
            新创建的子节点
        """
        # 执行走法创建新棋盘
        new_board = self.board.copy()
        new_board.make_move(move)
        
        # 创建子节点
        child = MCTSNode(board=new_board, parent=self, prior=prior)
        self.children[move] = child
        return child
    
    def best_child(self, c_puct: float = 1.5) -> Optional['MCTSNode']:
        """
        使用 PUCT 公式选择最佳子节点
        
        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: 探索系数
            
        Returns:
            最佳子节点
        """
        if not self.children:
            return None
        
        best_score = -float('inf')
        best_node = None
        
        for move, child in self.children.items():
            # PUCT 公式
            exploitation = child.Q
            exploration = c_puct * child.P * np.sqrt(self.N) / (1 + child.N)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_node = child
        
        return best_node
    
    def __repr__(self) -> str:
        return f"MCTSNode(Q={self.Q:.3f}, N={self.N}, P={self.P:.3f})"
