"""
单盘对局生命周期管理
setup → play → end
"""

import asyncio
from typing import Optional, Callable, Awaitable, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import chess

from .board import Board
from .rules import RuleChecker, MoveValidator, MoveResult, MoveStatus


class GameState(Enum):
    """游戏状态枚举"""
    PENDING = "pending"          # 等待开始
    PLAYING = "playing"          # 进行中
    PAUSED = "paused"            # 已暂停
    COMPLETED = "completed"      # 已完成
    ABORTED = "aborted"          # 已中止


@dataclass
class Player:
    """玩家信息"""
    agent_id: str
    color: bool  # True=白方，False=黑方
    is_human: bool = False


@dataclass
class GameResult:
    """游戏结果"""
    winner: Optional[str]         # 获胜者 ID，和棋为 None
    result_code: str              # "1-0", "0-1", "1/2-1/2"
    reason: str                   # 结束原因
    move_count: int               # 总步数
    pgn: str                      # PGN 记录


class Game:
    """
    单盘对局类
    管理一局完整的游戏生命周期
    """
    
    def __init__(
        self,
        game_id: str,
        white_player: Player,
        black_player: Player,
        validator: Optional[MoveValidator] = None,
        on_move_callback: Optional[Callable[[chess.Move], None]] = None,
        on_end_callback: Optional[Callable[['GameResult'], None]] = None,
    ):
        """
        初始化对局
        
        Args:
            game_id: 对局唯一标识
            white_player: 白方玩家
            black_player: 黑方玩家
            validator: 走法验证器
            on_move_callback: 走子回调函数
            on_end_callback: 游戏结束回调函数
        """
        self.game_id = game_id
        self.white_player = white_player
        self.black_player = black_player
        self.board = Board()
        self.validator = validator or MoveValidator()
        self.on_move_callback = on_move_callback
        self.on_end_callback = on_end_callback
        
        self.state = GameState.PENDING
        self._move_history: List[chess.Move] = []
        self._pgn_moves: List[str] = []
        self._current_turn_color: bool = chess.WHITE
    
    @property
    def current_player(self) -> Player:
        """获取当前回合玩家"""
        if self.board.turn == chess.WHITE:
            return self.white_player
        else:
            return self.black_player
    
    @property
    def move_count(self) -> int:
        """获取已走步数"""
        return len(self._move_history)
    
    def start(self) -> bool:
        """
        开始游戏
        
        Returns:
            bool: 是否成功开始
        """
        if self.state != GameState.PENDING:
            return False
        
        self.state = GameState.PLAYING
        self.board.reset()
        self._move_history.clear()
        self._pgn_moves.clear()
        self._current_turn_color = chess.WHITE
        return True
    
    def pause(self) -> bool:
        """暂停游戏"""
        if self.state != GameState.PLAYING:
            return False
        self.state = GameState.PAUSED
        return True
    
    def resume(self) -> bool:
        """恢复游戏"""
        if self.state != GameState.PAUSED:
            return False
        self.state = GameState.PLAYING
        return True
    
    async def make_move(
        self,
        agent_id: str,
        proposed_move: str
    ) -> Tuple[MoveResult, bool]:
        """
        执行一步棋（异步）
        
        Args:
            agent_id: 智能体 ID
            proposed_move: 提议的走法
            
        Returns:
            Tuple[MoveResult, bool]: (验证结果，是否成功)
        """
        if self.state != GameState.PLAYING:
            return MoveResult(
                status=MoveStatus.GAME_OVER if self.state == GameState.COMPLETED else MoveStatus.INVALID_FORMAT,
                success=False,
                reason=f"Game not in playing state: {self.state.value}"
            ), False
        
        # 检查是否是当前回合的玩家
        current_player = self.current_player
        if agent_id != current_player.agent_id:
            return MoveResult(
                status=MoveStatus.WRONG_TURN,
                success=False,
                reason=f"Not your turn. Current player: {current_player.agent_id}"
            ), False
        
        # 开始计时
        self.validator.start_move_timer(agent_id)
        
        # 验证并执行走法
        result, success = self.validator.validate_and_execute(
            agent_id,
            proposed_move,
            self.board.internal_board
        )
        
        if success and result.move:
            self._move_history.append(result.move)
            # 使用 UCI 格式记录走法（简化处理）
            self._pgn_moves.append(result.move.uci())
            
            # 调用回调
            if self.on_move_callback:
                self.on_move_callback(result.move)
            
            # 检查游戏是否结束
            self._check_game_end()
        
        return result, success
    
    def _check_game_end(self) -> None:
        """检查游戏是否结束"""
        if not self.board.is_game_over:
            return
        
        # 游戏结束
        rule_checker = RuleChecker()
        result_code = rule_checker.get_result(self.board.internal_board)
        reason = rule_checker.get_game_end_reason(self.board.internal_board) or "game_over"
        
        # 确定获胜者
        winner = None
        if result_code == "1-0":
            winner = self.white_player.agent_id
        elif result_code == "0-1":
            winner = self.black_player.agent_id
        # 和棋 winner 为 None
        
        game_result = GameResult(
            winner=winner,
            result_code=result_code,
            reason=reason,
            move_count=self.move_count,
            pgn=self._generate_pgn(result_code)
        )
        
        self.state = GameState.COMPLETED
        
        # 调用结束回调
        if self.on_end_callback:
            self.on_end_callback(game_result)
    
    def _generate_pgn(self, result_code: str) -> str:
        """生成 PGN 格式记录"""
        pgn = f"[Event \"ChessRL Game {self.game_id}\"]\n"
        pgn += f"[White \"{self.white_player.agent_id}\"]\n"
        pgn += f"[Black \"{self.black_player.agent_id}\"]\n"
        pgn += f"[Result \"{result_code}\"]\n"
        pgn += f"[FEN \"{self.board.fen}\"]\n\n"
        
        # 添加走法
        moves_str = " ".join(self._pgn_moves)
        pgn += moves_str
        
        return pgn
    
    def abort(self) -> None:
        """中止游戏"""
        self.state = GameState.ABORTED
    
    def get_pgn(self) -> str:
        """获取当前 PGN 记录"""
        result_code = RuleChecker().get_result(self.board.internal_board)
        if self.state == GameState.COMPLETED:
            return self._generate_pgn(result_code)
        
        # 游戏中
        pgn = f"[Event \"ChessRL Game {self.game_id}\"]\n"
        pgn += f"[White \"{self.white_player.agent_id}\"]\n"
        pgn += f"[Black \"{self.black_player.agent_id}\"]\n"
        pgn += f"[Result \"*\"]\n"
        pgn += f"[FEN \"{self.board.fen}\"]\n\n"
        pgn += " ".join(self._pgn_moves)
        return pgn
    
    def get_state_snapshot(self) -> dict:
        """获取游戏状态快照"""
        return {
            'game_id': self.game_id,
            'state': self.state.value,
            'fen': self.board.fen,
            'move_count': self.move_count,
            'white_player': self.white_player.agent_id,
            'black_player': self.black_player.agent_id,
            'current_turn': 'white' if self.board.turn else 'black',
            'is_check': self.board.get_state().is_check,
            'is_checkmate': self.board.get_state().is_checkmate,
        }
    
    async def play_full_game(
        self,
        white_move_func: Callable[[Board], Awaitable[str]],
        black_move_func: Callable[[Board], Awaitable[str]],
        delay_between_moves: float = 0.1
    ) -> GameResult:
        """
        自动完成整局游戏（用于测试/演示）
        
        Args:
            white_move_func: 白方走法函数 (接收 Board, 返回 UCI/SAN 走法)
            black_move_func: 黑方走法函数
            delay_between_moves: 每步之间的延迟（秒）
            
        Returns:
            GameResult: 游戏结果
        """
        self.start()
        
        result_future = asyncio.Future()
        
        def on_end_callback(result: GameResult):
            if not result_future.done():
                result_future.set_result(result)
        
        self.on_end_callback = on_end_callback
        
        while self.state == GameState.PLAYING:
            current_player = self.current_player
            
            # 获取走法
            if current_player.color == chess.WHITE:
                board_copy = self.board.copy()
                move = await white_move_func(board_copy)
            else:
                board_copy = self.board.copy()
                move = await black_move_func(board_copy)
            
            # 执行走法
            await self.make_move(current_player.agent_id, move)
            
            # 延迟
            if delay_between_moves > 0:
                await asyncio.sleep(delay_between_moves)
        
        return await result_future
