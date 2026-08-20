"""
棋盘状态封装 - python-chess wrapper
"""

import chess
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BoardState:
    """棋盘状态快照"""
    fen: str                          # FEN 字符串
    move_count: int                   # 总步数
    is_check: bool                    # 是否将军
    is_checkmate: bool                # 是否将死
    is_stalemate: bool                # 是否逼和
    is_insufficient_material: bool    # 是否子力不足
    is_fifty_moves: bool              # 是否 50 步规则
    is_repetition: bool               # 是否三次重复
    turn: bool                        # 当前回合 (True=白方)
    
    def to_dict(self) -> dict:
        return {
            'fen': self.fen,
            'move_count': self.move_count,
            'is_check': self.is_check,
            'is_checkmate': self.is_checkmate,
            'is_stalemate': self.is_stalemate,
            'is_insufficient_material': self.is_insufficient_material,
            'is_fifty_moves': self.is_fifty_moves,
            'is_repetition': self.is_repetition,
            'turn': self.turn,
        }


class Board:
    """
    棋盘封装类
    基于 python-chess 库，提供棋盘状态管理和走法执行
    """
    
    def __init__(self, fen: Optional[str] = None):
        """
        初始化棋盘
        
        Args:
            fen: 初始 FEN 字符串，默认为标准开局位置
        """
        if fen is None:
            self._board = chess.Board()
        else:
            self._board = chess.Board(fen)
        
        self._move_history: List[chess.Move] = []
    
    @property
    def internal_board(self) -> chess.Board:
        """返回内部 chess.Board 对象"""
        return self._board
    
    @property
    def fen(self) -> str:
        """获取当前 FEN 字符串"""
        return self._board.fen()
    
    @property
    def turn(self) -> bool:
        """获取当前回合 (True=白方，False=黑方)"""
        return self._board.turn
    
    @property
    def move_count(self) -> int:
        """获取已走步数"""
        return len(self._move_history)
    
    @property
    def legal_moves(self) -> List[chess.Move]:
        """获取所有合法走法"""
        return list(self._board.legal_moves)
    
    @property
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        return self._board.is_game_over()
    
    def get_state(self) -> BoardState:
        """获取当前棋盘状态快照"""
        return BoardState(
            fen=self._board.fen(),
            move_count=len(self._move_history),
            is_check=self._board.is_check(),
            is_checkmate=self._board.is_checkmate(),
            is_stalemate=self._board.is_stalemate(),
            is_insufficient_material=self._board.is_insufficient_material(),
            is_fifty_moves=self._board.is_fifty_moves(),
            is_repetition=self._board.can_claim_threefold_repetition(),
            turn=self._board.turn,
        )
    
    def make_move(self, move: chess.Move) -> bool:
        """
        执行一步棋
        
        Args:
            move: chess.Move 对象
            
        Returns:
            bool: 是否成功执行
        """
        if move not in self._board.legal_moves:
            return False
        
        self._board.push(move)
        self._move_history.append(move)
        return True
    
    def make_move_uci(self, uci_move: str) -> bool:
        """
        通过 UCI 格式执行走法
        
        Args:
            uci_move: UCI 格式字符串，如 "e2e4"
            
        Returns:
            bool: 是否成功执行
        """
        try:
            move = chess.Move.from_uci(uci_move)
            return self.make_move(move)
        except ValueError:
            return False
    
    def make_move_san(self, san_move: str) -> bool:
        """
        通过 SAN 格式执行走法
        
        Args:
            san_move: SAN 格式字符串，如 "Nf3", "O-O"
            
        Returns:
            bool: 是否成功执行
        """
        try:
            move = self._board.parse_san(san_move)
            return self.make_move(move)
        except ValueError:
            return False
    
    def undo(self) -> Optional[chess.Move]:
        """
        撤销最后一步
        
        Returns:
            被撤销的走法，如果无步可撤则返回 None
        """
        if self._move_history:
            move = self._board.pop()
            self._move_history.pop()
            return move
        return None
    
    def reset(self) -> None:
        """重置棋盘到初始状态"""
        self._board.reset()
        self._move_history.clear()
    
    def copy(self) -> 'Board':
        """创建棋盘副本"""
        new_board = Board(self.fen)
        new_board._move_history = self._move_history.copy()
        return new_board
    
    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """获取指定位置的棋子"""
        return self._board.piece_at(square)
    
    def get_piece_at_coords(self, file: int, rank: int) -> Optional[chess.Piece]:
        """
        获取指定坐标的棋子
        
        Args:
            file: 文件 (0-7, a-h)
            rank: 秩 (0-7, 1-8)
        """
        square = chess.square(file, rank)
        return self._board.piece_at(square)
    
    def __str__(self) -> str:
        return self._board.unicode()
