"""
Replay Buffer - 经验回放缓冲区

存储单元：(board_state_tensor, action_index, π_mcts_vector, reward_z, game_id, round_id)
- 容量：REPLAY_BUFFER_SIZE (初始 500 局)
- 采样：按 game_id 均匀采样（避免同局相关性）
- 淘汰：FIFO，最旧先出
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import random


@dataclass
class StepRecord:
    """单步记录"""
    board_state: torch.Tensor      # [16, 8, 8] 棋盘状态
    action_index: int              # 走法索引 (0-4607)
    mcts_policy: np.ndarray        # [4608] MCTS 策略分布
    reward_z: float                # 游戏结果 (-1, 0, 1)
    game_id: str                   # 游戏 ID
    round_id: int                  # 轮次 ID


@dataclass
class Batch:
    """采样批次"""
    states: torch.Tensor           # [B, 16, 8, 8]
    actions: torch.Tensor          # [B]
    mcts_policies: torch.Tensor    # [B, 4608]
    rewards: torch.Tensor          # [B, 1]
    game_ids: List[str]


class ReplayBuffer:
    """
    经验回放缓冲区
    
    特性：
    - FIFO 淘汰策略
    - 按游戏 ID 分组存储
    - 支持均匀采样（避免同局相关性）
    """
    
    def __init__(self, capacity: int = 500):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量（局数）
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
        # 按游戏 ID 索引
        self.game_indices: Dict[str, List[int]] = {}
        
        # 统计信息
        self.total_steps = 0
        self.total_games = 0
    
    def push(self, trajectory: List[StepRecord]) -> None:
        """
        添加一条完整的游戏轨迹
        
        Args:
            trajectory: 游戏的所有步记录列表
        """
        if not trajectory:
            return
        
        game_id = trajectory[0].game_id
        
        # 如果游戏已满，移除旧数据
        if game_id in self.game_indices:
            old_indices = self.game_indices[game_id]
            # 标记为待删除（实际由 deque 自动处理）
            pass
        
        # 添加新数据
        start_idx = len(self.buffer)
        for step in trajectory:
            self.buffer.append(step)
            self.total_steps += 1
        
        # 更新游戏索引
        new_indices = list(range(start_idx, len(self.buffer)))
        self.game_indices[game_id] = new_indices
        self.total_games += 1
        
        # 清理过期索引（超出容量的游戏）
        self._cleanup_indices()
    
    def _cleanup_indices(self) -> None:
        """清理超出容量的游戏索引"""
        # 简单实现：当 buffer 满时，重建索引
        if len(self.buffer) == self.capacity:
            # 获取当前 buffer 中有效的游戏 ID
            valid_game_ids = set(step.game_id for step in self.buffer)
            
            # 删除无效索引
            invalid_keys = [k for k in self.game_indices if k not in valid_game_ids]
            for key in invalid_keys:
                del self.game_indices[key]
    
    def sample(self, batch_size: int) -> Optional[Batch]:
        """
        随机采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Batch 对象，如果 buffer 为空则返回 None
        """
        if len(self.buffer) < batch_size:
            return None
        
        # 按游戏 ID 均匀采样（避免同局强相关性）
        sampled_steps = []
        game_ids_sampled = []
        
        # 策略 1：完全随机采样
        # indices = random.sample(range(len(self.buffer)), batch_size)
        # sampled_steps = [self.buffer[i] for i in indices]
        
        # 策略 2：按游戏采样（推荐）
        available_games = list(self.game_indices.keys())
        if not available_games:
            return None
        
        # 每个游戏采样的步数
        steps_per_game = max(1, batch_size // len(available_games))
        
        for game_id in available_games:
            indices = self.game_indices[game_id]
            if not indices:
                continue
            
            # 从该游戏中采样
            n_samples = min(steps_per_game, len(indices))
            sampled_indices = random.sample(indices, n_samples)
            
            for idx in sampled_indices:
                sampled_steps.append(self.buffer[idx])
                game_ids_sampled.append(game_id)
            
            if len(sampled_steps) >= batch_size:
                break
        
        # 如果还不够，继续随机采样
        while len(sampled_steps) < batch_size:
            idx = random.randint(0, len(self.buffer) - 1)
            sampled_steps.append(self.buffer[idx])
            game_ids_sampled.append(self.buffer[idx].game_id)
        
        # 构建 Batch
        states = torch.stack([s.board_state for s in sampled_steps[:batch_size]])
        actions = torch.tensor([s.action_index for s in sampled_steps[:batch_size]], dtype=torch.long)
        mcts_policies = torch.from_numpy(np.array([s.mcts_policy for s in sampled_steps[:batch_size]]).astype(np.float32))
        rewards = torch.tensor([[s.reward_z] for s in sampled_steps[:batch_size]], dtype=torch.float32)
        
        return Batch(
            states=states,
            actions=actions,
            mcts_policies=mcts_policies,
            rewards=rewards,
            game_ids=game_ids_sampled[:batch_size]
        )
    
    def sample_by_game(self, n_games: int, steps_per_game: int = 1) -> Optional[Batch]:
        """
        按游戏采样（确保来自不同游戏）
        
        Args:
            n_games: 采样游戏数量
            steps_per_game: 每个游戏采样的步数
            
        Returns:
            Batch 对象
        """
        available_games = list(self.game_indices.keys())
        if len(available_games) < n_games:
            return None
        
        # 随机选择游戏
        selected_games = random.sample(available_games, n_games)
        
        sampled_steps = []
        game_ids_sampled = []
        
        for game_id in selected_games:
            indices = self.game_indices[game_id]
            if not indices:
                continue
            
            n_samples = min(steps_per_game, len(indices))
            sampled_indices = random.sample(indices, n_samples)
            
            for idx in sampled_indices:
                sampled_steps.append(self.buffer[idx])
                game_ids_sampled.append(game_id)
        
        if not sampled_steps:
            return None
        
        # 构建 Batch
        states = torch.stack([s.board_state for s in sampled_steps])
        actions = torch.tensor([s.action_index for s in sampled_steps], dtype=torch.long)
        mcts_policies = torch.from_numpy(np.array([s.mcts_policy for s in sampled_steps]).astype(np.float32))
        rewards = torch.tensor([[s.reward_z] for s in sampled_steps], dtype=torch.float32)
        
        return Batch(
            states=states,
            actions=actions,
            mcts_policies=mcts_policies,
            rewards=rewards,
            game_ids=game_ids_sampled
        )
    
    def __len__(self) -> int:
        """返回缓冲区中的总步数"""
        return len(self.buffer)
    
    def n_games(self) -> int:
        """返回缓冲区中的游戏数量"""
        return len(self.game_indices)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        self.game_indices.clear()
        self.total_steps = 0
        self.total_games = 0
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_steps': self.total_steps,
            'total_games': self.total_games,
            'buffer_size': len(self.buffer),
            'capacity': self.capacity,
            'n_games': len(self.game_indices),
        }


if __name__ == '__main__':
    # 测试 ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    
    # 创建模拟数据
    for game_idx in range(10):
        trajectory = []
        for step_idx in range(20):
            step = StepRecord(
                board_state=torch.randn(16, 8, 8),
                action_index=random.randint(0, 4607),
                mcts_policy=np.random.rand(4608).astype(np.float32),
                reward_z=random.choice([-1, 0, 1]),
                game_id=f"game_{game_idx}",
                round_id=game_idx
            )
            trajectory.append(step)
        
        buffer.push(trajectory)
    
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # 采样测试
    batch = buffer.sample(batch_size=32)
    if batch:
        print(f"Sampled batch: states={batch.states.shape}, actions={batch.actions.shape}")
        print(f"MCTS policies: {batch.mcts_policies.shape}, rewards: {batch.rewards.shape}")
