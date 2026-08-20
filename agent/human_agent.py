"""
Human Agent - 人类玩家交互代理
"""

import asyncio
from typing import Optional, Tuple
import chess

from engine.board import Board
from .base import Agent, AgentType


class HumanAgent(Agent):
    """
    人类玩家智能体
    
    通过 UI 交互获取走法输入
    支持鼠标点击选择棋盘格子
    """
    
    def __init__(self, agent_id: str, color: chess.Color = chess.WHITE):
        """
        初始化人类玩家
        
        Args:
            agent_id: 智能体 ID
            color: 玩家执棋颜色（白/黑）
        """
        super().__init__(agent_id, AgentType.HUMAN)
        
        self.color = color
        self._pending_move: Optional[int] = None
        self._move_event: Optional[asyncio.Event] = None
        self._selected_square: Optional[int] = None
    
    async def think(self, board: Board) -> str:
        """
        等待人类玩家输入走法
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            str: UCI 格式走法
        """
        # 创建等待事件
        self._move_event = asyncio.Event()
        self._pending_move = None
        self._selected_square = None
        
        # 等待玩家输入
        await self._move_event.wait()
        
        # 获取选中的起始格和目标格
        if self._pending_move is not None:
            from_square, to_square = self._pending_move
            move = chess.Move(from_square, to_square)
            
            # 检查是否为升变走法，默认升变为后
            piece = board.internal_board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if (board.internal_board.turn == chess.WHITE and to_square >= 56) or \
                   (board.internal_board.turn == chess.BLACK and to_square <= 7):
                    # 升变，默认为后
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            
            # 验证走法合法性
            if move in board.legal_moves:
                return move.uci()
            else:
                # 非法走法，返回空
                return ""
        
        return ""
    
    def on_move_request(self, from_square: int, to_square: int) -> None:
        """
        处理玩家走法请求（由 UI 调用）
        
        Args:
            from_square: 起始格子 (0-63)
            to_square: 目标格子 (0-63)
        """
        self._pending_move = (from_square, to_square)
        if self._move_event:
            self._move_event.set()
    
    def on_square_selected(self, square: int) -> None:
        """
        处理格子选中事件（用于高亮显示）
        
        Args:
            square: 选中的格子 (0-63)
        """
        self._selected_square = square
    
    def reset(self) -> None:
        """重置玩家状态（每局新游戏时调用）"""
        self._pending_move = None
        self._move_event = None
        self._selected_square = None
    
    def get_selected_square(self) -> Optional[int]:
        """获取当前选中的格子"""
        return self._selected_square
    
    def clear_selection(self) -> None:
        """清除选中状态"""
        self._selected_square = None
    
    def __str__(self) -> str:
        return f"Human({self.agent_id})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} color={'white' if self.color else 'black'}>"


def create_human_agent(agent_id: str, color: chess.Color = chess.WHITE) -> HumanAgent:
    """
    创建人类玩家的工厂函数
    
    Args:
        agent_id: 智能体 ID
        color: 玩家执棋颜色
        
    Returns:
        HumanAgent 实例
    """
    return HumanAgent(agent_id=agent_id, color=color)
