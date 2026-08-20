"""
棋盘状态管理
提供实时棋盘状态查询和更新
"""

from typing import Dict, List, Optional
import chess

from engine.board import Board, BoardState


class BoardStateManager:
    """
    棋盘状态管理器
    管理多个游戏的棋盘状态，提供统一访问接口
    """
    
    def __init__(self):
        """初始化状态管理器"""
        self._boards: Dict[str, Board] = {}  # game_id -> Board
        self._states: Dict[str, BoardState] = {}  # game_id -> BoardState
    
    def register_board(self, game_id: str, board: Board) -> None:
        """
        注册棋盘
        
        Args:
            game_id: 游戏 ID
            board: Board 对象
        """
        self._boards[game_id] = board
        self._states[game_id] = board.get_state()
    
    def unregister_board(self, game_id: str) -> None:
        """注销棋盘"""
        if game_id in self._boards:
            del self._boards[game_id]
        if game_id in self._states:
            del self._states[game_id]
    
    def update_state(self, game_id: str) -> Optional[BoardState]:
        """
        更新指定游戏的棋盘状态
        
        Args:
            game_id: 游戏 ID
            
        Returns:
            更新后的 BoardState，如果 game_id 不存在则返回 None
        """
        if game_id not in self._boards:
            return None
        
        board = self._boards[game_id]
        self._states[game_id] = board.get_state()
        return self._states[game_id]
    
    def get_state(self, game_id: str) -> Optional[BoardState]:
        """获取指定游戏的棋盘状态"""
        return self._states.get(game_id)
    
    def get_fen(self, game_id: str) -> Optional[str]:
        """获取指定游戏的 FEN 字符串"""
        if game_id in self._boards:
            return self._boards[game_id].fen
        return None
    
    def get_all_states(self) -> Dict[str, BoardState]:
        """获取所有游戏的状态"""
        return self._states.copy()
    
    def get_all_fens(self) -> Dict[str, str]:
        """获取所有游戏的 FEN"""
        return {
            game_id: board.fen
            for game_id, board in self._boards.items()
        }
    
    def is_game_over(self, game_id: str) -> bool:
        """检查游戏是否结束"""
        if game_id in self._boards:
            return self._boards[game_id].is_game_over
        return False
    
    def get_legal_moves(self, game_id: str) -> List[chess.Move]:
        """获取合法走法列表"""
        if game_id in self._boards:
            return self._boards[game_id].legal_moves
        return []
    
    def clear(self) -> None:
        """清空所有状态"""
        self._boards.clear()
        self._states.clear()
