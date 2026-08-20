"""
匹配、并发调度、角色分配
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import random

from .game import Game, Player, GameState, GameResult
from .scoring import ScoringSystem


@dataclass
class MatchConfig:
    """对局配置"""
    max_concurrent_games: int = 4  # 最大并发对局数
    timeout_per_move: float = 120.0  # 每步超时（秒）


class Scheduler:
    """
    调度器基类
    """
    
    def __init__(self, config: Optional[MatchConfig] = None):
        self.config = config or MatchConfig()
        self._active_games: Dict[str, Game] = {}
        self._completed_games: List[Game] = []


class MatchScheduler(Scheduler):
    """
    匹配调度器
    负责智能体配对、并发控制、游戏生命周期管理
    """
    
    def __init__(
        self,
        scoring_system: ScoringSystem,
        config: Optional[MatchConfig] = None,
        on_game_end_callback: Optional[Callable[[Game], None]] = None,
    ):
        """
        初始化匹配调度器
        
        Args:
            scoring_system: 计分系统
            config: 对局配置
            on_game_end_callback: 游戏结束回调
        """
        super().__init__(config)
        self.scoring_system = scoring_system
        self.on_game_end_callback = on_game_end_callback
        self._agent_pool: List[str] = []
        self._busy_agents: set = set()
        self._game_counter = 0
        self._lock = asyncio.Lock()
    
    def register_agents(self, agent_ids: List[str]) -> None:
        """注册智能体池"""
        self._agent_pool = agent_ids.copy()
        for agent_id in agent_ids:
            self.scoring_system.register_agent(agent_id)
    
    def _generate_game_id(self) -> str:
        """生成游戏 ID"""
        self._game_counter += 1
        return f"game_{self._game_counter:04d}"
    
    def _get_available_agents(self) -> List[str]:
        """获取空闲智能体"""
        return [
            agent_id for agent_id in self._agent_pool
            if agent_id not in self._busy_agents
        ]
    
    def _pair_agents_random(self) -> List[Tuple[str, str]]:
        """
        随机配对智能体
        
        Returns:
            List[Tuple[white_agent, black_agent]]: 配对列表
        """
        available = self._get_available_agents()
        
        # 确保偶数个智能体
        if len(available) % 2 != 0:
            # 如果奇数，随机移除一个
            available.pop(random.randint(0, len(available) - 1))
        
        # 随机打乱后配对
        random.shuffle(available)
        
        pairs = []
        for i in range(0, len(available), 2):
            white = available[i]
            black = available[i + 1]
            pairs.append((white, black))
        
        return pairs
    
    def _pair_agents_ranked(self) -> List[Tuple[str, str]]:
        """
        按排名配对（第 1 vs 第 2, 第 3 vs 第 4, ...）
        
        Returns:
            List[Tuple[white_agent, black_agent]]: 配对列表
        """
        rankings = self.scoring_system.get_rankings()
        
        if len(rankings) < 2:
            return []
        
        # 确保偶数个
        if len(rankings) % 2 != 0:
            rankings = rankings[:-1]
        
        pairs = []
        for i in range(0, len(rankings), 2):
            # 高排名执白
            white = rankings[i][0]
            black = rankings[i + 1][0]
            pairs.append((white, black))
        
        return pairs
    
    async def create_game(
        self,
        white_agent: str,
        black_agent: str,
        move_callback: Optional[Callable] = None,
    ) -> Optional[Game]:
        """
        创建一局游戏
        
        Args:
            white_agent: 白方智能体 ID
            black_agent: 黑方智能体 ID
            move_callback: 走子回调函数
            
        Returns:
            Game: 游戏对象，如果创建失败则返回 None
        """
        async with self._lock:
            # 检查智能体是否空闲
            if white_agent in self._busy_agents or black_agent in self._busy_agents:
                return None
            
            # 标记为忙碌
            self._busy_agents.add(white_agent)
            self._busy_agents.add(black_agent)
            
            # 创建玩家
            white_player = Player(agent_id=white_agent, color=True)
            black_player = Player(agent_id=black_agent, color=False)
            
            # 创建游戏
            game_id = self._generate_game_id()
            
            def on_end_callback(result: GameResult):
                self._on_game_end(game_id, result)
            
            game = Game(
                game_id=game_id,
                white_player=white_player,
                black_player=black_player,
                on_move_callback=move_callback,
                on_end_callback=on_end_callback,
            )
            
            self._active_games[game_id] = game
            
            return game
    
    def _on_game_end(self, game_id: str, result: GameResult) -> None:
        """游戏结束处理"""
        if game_id not in self._active_games:
            return
        
        game = self._active_games[game_id]
        
        # 释放智能体
        self._busy_agents.discard(game.white_player.agent_id)
        self._busy_agents.discard(game.black_player.agent_id)
        
        # 记录分数
        self.scoring_system.record_game(
            game_id=game_id,
            white_agent=game.white_player.agent_id,
            black_agent=game.black_player.agent_id,
            result_code=result.result_code,
        )
        
        # 移动到已完成列表
        self._completed_games.append(game)
        del self._active_games[game_id]
        
        # 调用回调
        if self.on_game_end_callback:
            self.on_game_end_callback(game)
    
    async def start_match_round(self, use_ranking: bool = False) -> List[Game]:
        """
        开始一轮匹配
        
        Args:
            use_ranking: 是否使用排名配对（否则随机）
            
        Returns:
            List[Game]: 创建的游戏列表
        """
        games = []
        
        # 配对
        if use_ranking:
            pairs = self._pair_agents_ranked()
        else:
            pairs = self._pair_agents_random()
        
        # 创建游戏（不超过最大并发数）
        max_games = min(len(pairs), self.config.max_concurrent_games)
        
        for i in range(max_games):
            white, black = pairs[i]
            game = await self.create_game(white, black)
            if game:
                game.start()
                games.append(game)
        
        return games
    
    def get_active_games(self) -> List[Game]:
        """获取所有进行中的游戏"""
        return list(self._active_games.values())
    
    def get_completed_games(self) -> List[Game]:
        """获取所有已完成的游戏"""
        return self._completed_games.copy()
    
    def get_game_status(self, game_id: str) -> Optional[dict]:
        """获取指定游戏的状态"""
        if game_id in self._active_games:
            return self._active_games[game_id].get_state_snapshot()
        return None
    
    def get_all_games_status(self) -> List[dict]:
        """获取所有游戏的状态"""
        status_list = []
        for game in self._active_games.values():
            status_list.append(game.get_state_snapshot())
        return status_list
    
    def abort_game(self, game_id: str) -> bool:
        """中止指定游戏"""
        if game_id not in self._active_games:
            return False
        
        game = self._active_games[game_id]
        game.abort()
        
        # 释放智能体
        self._busy_agents.discard(game.white_player.agent_id)
        self._busy_agents.discard(game.black_player.agent_id)
        
        del self._active_games[game_id]
        return True
    
    def reset(self) -> None:
        """重置调度器"""
        self._active_games.clear()
        self._completed_games.clear()
        self._busy_agents.clear()
        self._game_counter = 0
