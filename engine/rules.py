"""
走法合法性、超时、和局、终局判定规则
"""

import time
import chess
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class MoveStatus(Enum):
    """走法状态枚举"""
    VALID = "valid"                      # 合法
    INVALID_FORMAT = "invalid_format"    # 格式错误
    ILLEGAL_MOVE = "illegal_move"        # 非法走法
    TIMEOUT = "timeout"                  # 超时
    GAME_OVER = "game_over"              # 游戏已结束
    WRONG_TURN = "wrong_turn"            # 不是当前回合方


@dataclass
class MoveResult:
    """走法校验结果"""
    status: MoveStatus
    success: bool
    reason: str = ""
    penalty: Optional[str] = None       # 违规处罚（如有）
    move: Optional[chess.Move] = None   # 解析后的走法对象


class RuleChecker:
    """
    规则检查器
    处理国际象棋各种终局条件
    """
    
    @staticmethod
    def is_checkmate(board: chess.Board) -> bool:
        """检查是否将死"""
        return board.is_checkmate()
    
    @staticmethod
    def is_stalemate(board: chess.Board) -> bool:
        """检查是否逼和（无子可动但不是将军）"""
        return board.is_stalemate()
    
    @staticmethod
    def is_insufficient_material(board: chess.Board) -> bool:
        """检查是否子力不足（无法将死）"""
        return board.is_insufficient_material()
    
    @staticmethod
    def is_fifty_moves(board: chess.Board) -> bool:
        """检查是否触发 50 步规则"""
        return board.is_fifty_moves()
    
    @staticmethod
    def is_threefold_repetition(board: chess.Board) -> bool:
        """检查是否三次重复局面"""
        return board.can_claim_threefold_repetition()
    
    @staticmethod
    def is_seventyfive_moves(board: chess.Board) -> bool:
        """检查是否 75 步规则（自动判和）"""
        return board.is_seventyfive_moves()
    
    @staticmethod
    def is_fivefold_repetition(board: chess.Board) -> bool:
        """检查是否五次重复局面（自动判和）"""
        return board.is_fivefold_repetition()
    
    @classmethod
    def get_game_end_reason(cls, board: chess.Board) -> Optional[str]:
        """
        获取游戏结束原因
        
        Returns:
            结束原因字符串，如果游戏未结束则返回 None
        """
        if board.is_checkmate():
            return "checkmate"
        elif board.is_stalemate():
            return "stalemate"
        elif board.is_insufficient_material():
            return "insufficient_material"
        elif board.is_fifty_moves():
            return "fifty_moves"
        elif board.can_claim_threefold_repetition():
            return "threefold_repetition"
        elif board.is_seventyfive_moves():
            return "seventyfive_moves"
        elif board.is_fivefold_repetition():
            return "fivefold_repetition"
        
        return None
    
    @classmethod
    def get_result(cls, board: chess.Board) -> str:
        """
        获取游戏结果
        
        Returns:
            "1-0" (白胜), "0-1" (黑胜), "1/2-1/2" (和棋), "*" (未结束)
        """
        if not board.is_game_over():
            return "*"
        
        if board.is_checkmate():
            return "1-0" if board.turn == chess.BLACK else "0-1"
        
        return "1/2-1/2"


class MoveValidator:
    """
    走法验证器
    验证走法的合法性和处理违规情况
    """
    
    def __init__(self, timeout_seconds: float = 120.0):
        """
        初始化验证器
        
        Args:
            timeout_seconds: 每步思考超时时间（秒）
        """
        self.timeout_seconds = timeout_seconds
        self._move_start_times: dict = {}
    
    def set_timeout(self, timeout_seconds: float) -> None:
        """设置超时时间"""
        self.timeout_seconds = timeout_seconds
    
    def start_move_timer(self, agent_id: str) -> None:
        """开始计时"""
        self._move_start_times[agent_id] = time.time()
    
    def check_timeout(self, agent_id: str) -> bool:
        """
        检查是否超时
        
        Returns:
            bool: 是否已超时
        """
        if agent_id not in self._move_start_times:
            return False
        
        elapsed = time.time() - self._move_start_times[agent_id]
        return elapsed > self.timeout_seconds
    
    def validate(
        self,
        agent_id: str,
        proposed_move: str,
        board: chess.Board
    ) -> MoveResult:
        """
        完整验证走法
        
        Args:
            agent_id: 智能体 ID
            proposed_move: 提议的走法 (UCI 或 SAN 格式)
            board: 当前棋盘状态
            
        Returns:
            MoveResult: 验证结果
        """
        # 1. 检查游戏是否已结束
        if board.is_game_over():
            return MoveResult(
                status=MoveStatus.GAME_OVER,
                success=False,
                reason="Game has ended",
                penalty=None
            )
        
        # 2. 检查是否超时
        if self.check_timeout(agent_id):
            return MoveResult(
                status=MoveStatus.TIMEOUT,
                success=False,
                reason=f"Timeout after {self.timeout_seconds}s",
                penalty="loss"  # 超时判负
            )
        
        # 3. 尝试解析走法（先试 UCI，再试 SAN）
        move = None
        try:
            move = chess.Move.from_uci(proposed_move)
        except ValueError:
            try:
                move = board.parse_san(proposed_move)
            except ValueError:
                return MoveResult(
                    status=MoveStatus.INVALID_FORMAT,
                    success=False,
                    reason=f"Invalid move format: {proposed_move}"
                )
        
        # 4. 检查是否是当前回合方
        # 注意：这里假设 agent_id 包含颜色信息，或者通过其他方式判断
        # P0 阶段简化处理，不强制检查颜色匹配
        
        # 5. 检查走法是否合法
        if move not in board.legal_moves:
            return MoveResult(
                status=MoveStatus.ILLEGAL_MOVE,
                success=False,
                reason=f"Illegal move: {proposed_move}"
            )
        
        # 所有检查通过
        return MoveResult(
            status=MoveStatus.VALID,
            success=True,
            reason="Valid move",
            move=move
        )
    
    def validate_and_execute(
        self,
        agent_id: str,
        proposed_move: str,
        board: chess.Board
    ) -> Tuple[MoveResult, bool]:
        """
        验证并执行走法
        
        Returns:
            Tuple[MoveResult, bool]: (验证结果，是否成功执行)
        """
        result = self.validate(agent_id, proposed_move, board)
        
        if result.success and result.move:
            board.push(result.move)
            return result, True
        
        return result, False
    
    def reset_timer(self, agent_id: str) -> None:
        """重置指定智能体的计时器"""
        if agent_id in self._move_start_times:
            del self._move_start_times[agent_id]
    
    def clear_all_timers(self) -> None:
        """清除所有计时器"""
        self._move_start_times.clear()
