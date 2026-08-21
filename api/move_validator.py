"""
走法校验模块 - 验证走法的合法性和违规处理

负责检查：
1. 走法格式（SAN/UCI）
2. 是否在合法走法列表中
3. 超时检测
4. 重复走法（三次重复判和）
5. 50 步规则（无吃子/兵移动判和）
6. 终局判定（将死/逼和）
"""

import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import chess

from engine.board import Board


class MoveStatus(Enum):
    """走法状态"""
    VALID = "valid"               # 合法走法
    INVALID_FORMAT = "invalid_format"      # 格式错误
    ILLEGAL_MOVE = "illegal_move"         # 非法走法（不符合规则）
    TIMEOUT = "timeout"                   # 超时
    THREEFOLD_REPETITION = "threefold_repetition"  # 三次重复
    FIFTY_MOVE_RULE = "fifty_move_rule"          # 50 步规则
    CHECKMATE = "checkmate"                      # 将死
    STALEMATE = "stalemate"                      # 逼和
    INSUFFICIENT_MATERIAL = "insufficient_material"  # 子力不足


@dataclass
class MoveResult:
    """走法校验结果"""
    status: MoveStatus
    move: Optional[str] = None           # UCI 格式走法
    penalty: float = 0.0                 # 惩罚分数（如有违规）
    reason: str = ""                     # 详细说明
    is_terminal: bool = False            # 是否终局
    game_result: Optional[str] = None    # 终局结果 ("win", "loss", "draw")


class MoveValidator:
    """
    走法校验器
    
    验证智能体走法的合法性，处理各种违规情况
    """
    
    def __init__(self, timeout_seconds: float = 120.0):
        """
        初始化校验器
        
        Args:
            timeout_seconds: 每步思考超时（秒）
        """
        self.timeout_seconds = timeout_seconds
        self.move_history: Dict[str, list] = {}  # agent_id -> 历史走法
        self.position_history: Dict[str, list] = {}  # game_id -> 历史局面
    
    def validate(self, agent_id: str, proposed_move: str, 
                 board: Board, elapsed_time: float = 0.0) -> MoveResult:
        """
        验证走法
        
        Args:
            agent_id: 智能体 ID
            proposed_move: 提议的走法（UCI 或 SAN 格式）
            board: 当前棋盘状态
            elapsed_time: 已用时间（秒）
            
        Returns:
            MoveResult: 校验结果
        """
        # 1. 检查超时
        if elapsed_time > self.timeout_seconds:
            return MoveResult(
                status=MoveStatus.TIMEOUT,
                penalty=-1.0,
                reason=f"超时 ({elapsed_time:.2f}s > {self.timeout_seconds}s)",
                is_terminal=True,
                game_result="loss"
            )
        
        # 2. 检查走法格式
        parsed_move = self._parse_move(proposed_move, board)
        if parsed_move is None:
            return MoveResult(
                status=MoveStatus.INVALID_FORMAT,
                reason=f"无效走法格式：{proposed_move}",
                penalty=-0.5
            )
        
        # 3. 检查是否在合法走法列表中
        if parsed_move not in board.legal_moves:
            return MoveResult(
                status=MoveStatus.ILLEGAL_MOVE,
                reason=f"非法走法：{parsed_move.uci()}",
                penalty=-1.0
            )
        
        # 4. 执行走法并检查终局条件
        new_board = board.copy()
        new_board.push(parsed_move)
        
        # 检查将死
        if new_board.is_checkmate():
            return MoveResult(
                status=MoveStatus.CHECKMATE,
                move=parsed_move.uci(),
                is_terminal=True,
                game_result="win"
            )
        
        # 检查逼和
        if new_board.is_stalemate():
            return MoveResult(
                status=MoveStatus.STALEMATE,
                move=parsed_move.uci(),
                is_terminal=True,
                game_result="draw"
            )
        
        # 检查子力不足
        if new_board.is_insufficient_material():
            return MoveResult(
                status=MoveStatus.INSUFFICIENT_MATERIAL,
                move=parsed_move.uci(),
                is_terminal=True,
                game_result="draw"
            )
        
        # 5. 检查三次重复
        if self._is_threefold_repetition(new_board):
            return MoveResult(
                status=MoveStatus.THREEFOLD_REPETITION,
                move=parsed_move.uci(),
                is_terminal=True,
                game_result="draw"
            )
        
        # 6. 检查 50 步规则
        if new_board.is_fifty_moves():
            return MoveResult(
                status=MoveStatus.FIFTY_MOVE_RULE,
                move=parsed_move.uci(),
                is_terminal=True,
                game_result="draw"
            )
        
        # 所有检查通过
        return MoveResult(
            status=MoveStatus.VALID,
            move=parsed_move.uci()
        )
    
    def _parse_move(self, move_str: str, board: Board) -> Optional[chess.Move]:
        """
        解析走法字符串
        
        支持 UCI 和 SAN 格式
        
        Args:
            move_str: 走法字符串
            board: 棋盘状态
            
        Returns:
            chess.Move 或 None
        """
        if not move_str or not isinstance(move_str, str):
            return None
        
        move_str = move_str.strip()
        
        # 尝试 UCI 格式
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass
        
        # 尝试 SAN 格式
        try:
            move = board.parse_san(move_str)
            if move in board.legal_moves:
                return move
        except (ValueError, KeyError):
            pass
        
        return None
    
    def _is_threefold_repetition(self, board: Board) -> bool:
        """
        检查三次重复局面
        
        Args:
            board: 棋盘状态
            
        Returns:
            bool: 是否三次重复
        """
        return board.is_repetition(count=3)
    
    def record_move(self, agent_id: str, move: str, board_fen: str) -> None:
        """
        记录走法历史
        
        Args:
            agent_id: 智能体 ID
            move: UCI 格式走法
            board_fen: 局面 FEN 字符串
        """
        if agent_id not in self.move_history:
            self.move_history[agent_id] = []
        
        self.move_history[agent_id].append({
            'move': move,
            'fen': board_fen,
            'timestamp': time.time()
        })
    
    def record_position(self, game_id: str, board_fen: str) -> None:
        """
        记录局面历史（用于三次重复检测）
        
        Args:
            game_id: 游戏 ID
            board_fen: 局面 FEN 字符串
        """
        if game_id not in self.position_history:
            self.position_history[game_id] = []
        
        self.position_history[game_id].append(board_fen)
    
    def reset_agent_history(self, agent_id: str) -> None:
        """重置智能体走法历史"""
        if agent_id in self.move_history:
            self.move_history[agent_id] = []
    
    def reset_game_history(self, game_id: str) -> None:
        """重置游戏局面历史"""
        if game_id in self.position_history:
            self.position_history[game_id] = []
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        获取智能体统计信息
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            统计字典
        """
        history = self.move_history.get(agent_id, [])
        return {
            'total_moves': len(history),
            'avg_time_per_move': self._calculate_avg_time(history)
        }
    
    def _calculate_avg_time(self, history: list) -> float:
        """计算平均走法时间"""
        if len(history) < 2:
            return 0.0
        
        total_time = 0.0
        for i in range(1, len(history)):
            total_time += history[i]['timestamp'] - history[i-1]['timestamp']
        
        return total_time / (len(history) - 1)


# 便捷函数
def validate_move(agent_id: str, move: str, board: Board, 
                  timeout: float = 120.0) -> MoveResult:
    """
    便捷函数：验证走法
    
    Args:
        agent_id: 智能体 ID
        move: 走法字符串
        board: 棋盘状态
        timeout: 超时时间
        
    Returns:
        MoveResult: 校验结果
    """
    validator = MoveValidator(timeout_seconds=timeout)
    return validator.validate(agent_id, move, board)
