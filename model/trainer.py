"""
训练循环 - 蒸馏 + 自博弈训练

核心逻辑：
1. Student 用 MCTS 对 Teacher(冻结) 自博弈
2. 每步 (s, a, π_mcts, z) 写入 Replay Buffer
3. Loss = α·KL(π_T||π_S) + β·L_AZ + γ·||θ||²
4. 梯度只更新 Student
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import asyncio
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time

from .network import AlphaZeroResNet, create_model
from .loss import compute_loss, ModelOutput, MCTSTarget
from .replay_buffer import ReplayBuffer, StepRecord
from .encoder import encode_board

import chess


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    alpha: float = 1.0  # 蒸馏 loss 权重
    beta: float = 0.5   # 自博弈 loss 权重
    gamma: float = 0.01  # L2 正则权重
    replay_buffer_size: int = 500
    mcts_iterations: int = 50
    training_steps_per_game: int = 10  # 每局游戏后的训练步数
    checkpoint_interval: int = 100  # 检查点保存间隔
    log_interval: int = 10  # 日志输出间隔


class Trainer:
    """
    模型训练器
    
    负责：
    - 管理 Student 和 Teacher 模型
    - 执行自博弈生成训练数据
    - 从 Replay Buffer 采样训练
    - 定期评估和保存检查点
    """
    
    def __init__(self, config: TrainingConfig = None, device: str = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.config = config or TrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.student_model = create_model().to(self.device)
        self.teacher_model = create_model().to(self.device)
        
        # 冻结 Teacher 参数
        self._freeze_teacher()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.95
        )
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=self.config.replay_buffer_size)
        
        # TensorBoard 日志
        self.writer = SummaryWriter(log_dir='logs/training')
        
        # 统计信息
        self.global_step = 0
        self.games_trained = 0
        self.training_history = []
    
    def _freeze_teacher(self):
        """冻结 Teacher 模型参数"""
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def _unfreeze_student(self):
        """确保 Student 模型可训练"""
        for param in self.student_model.parameters():
            param.requires_grad = True
        self.student_model.train()
    
    async def self_play_game(self, game_id: str, round_id: int) -> List[StepRecord]:
        """
        执行一局自博弈（Student vs Teacher）
        
        Args:
            game_id: 游戏 ID
            round_id: 轮次 ID
            
        Returns:
            游戏轨迹（所有步的记录）
        """
        from mcts.search import mcts_search
        
        board = chess.Board()
        trajectory = []
        move_history = []
        
        # 导入 model_agent 用于实际走子
        from agent.model_agent import ModelAgent
        
        # 创建临时 agent
        student_agent = ModelAgent(
            model=self.student_model,
            agent_id="student",
            is_teacher=False,
            mcts_iterations=self.config.mcts_iterations,
            device=self.device
        )
        
        teacher_agent = ModelAgent(
            model=self.teacher_model,
            agent_id="teacher",
            is_teacher=True,
            mcts_iterations=self.config.mcts_iterations,
            device=self.device
        )
        
        max_moves = 200  # 防止无限循环
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            # 确定当前回合的 agent
            current_agent = student_agent if board.turn == chess.WHITE else teacher_agent
            
            # MCTS 搜索获取策略分布
            state_tensor = encode_board(board).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, value = current_agent.model(state_tensor)
                policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            
            # 简化：从合法走法中选择（实际应使用 MCTS）
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # 简单策略：从合法走法中随机选择（MCTS 应在 agent 内部调用）
            import random
            move = random.choice(legal_moves)
            
            # 记录步
            action_index = move.uci()  # 使用 UCI 字符串作为临时标识
            # 转换为索引（简化处理）
            action_idx = hash(action_index) % 4608
            
            move_history.append(move.uci())
            board.push(move)
            move_count += 1
        
        # 确定游戏结果
        result = 0.0  # 默认和棋
        if board.is_checkmate():
            # 将死：最后一步的玩家获胜
            result = 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.0
        elif board.is_fifty_moves() or board.is_repetition():
            result = 0.0
        
        # 构建轨迹（简化：为每个状态添加相同的最终结果）
        # 实际应该在每一步记录当时的状态和 MCTS 策略
        trajectory = []
        
        return trajectory, result, move_history
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        执行单步训练
        
        Args:
            batch: 从 Replay Buffer 采样的批次
            
        Returns:
            loss 字典
        """
        self._unfreeze_student()
        self.optimizer.zero_grad()
        
        # 准备数据
        states = batch.states.to(self.device)
        mcts_policies = batch.mcts_policies.to(self.device)
        rewards = batch.rewards.to(self.device)
        
        # Student 前向传播
        policy_logits, value = self.student_model(states)
        student_output = ModelOutput(policy_logits=policy_logits, value=value)
        
        # Teacher 前向传播（无梯度）
        with torch.no_grad():
            teacher_policy_logits, teacher_value = self.teacher_model(states)
            teacher_output = ModelOutput(
                policy_logits=teacher_policy_logits,
                value=teacher_value
            )
        
        # MCTS 目标
        mcts_target = MCTSTarget(policy=mcts_policies, value=rewards)
        
        # 计算 loss
        total_loss, loss_dict = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            student_model=self.student_model
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        
        # 优化步骤
        self.optimizer.step()
        
        # 更新全局步数
        self.global_step += 1
        
        # 记录日志
        if self.global_step % self.config.log_interval == 0:
            self._log_training(loss_dict)
        
        # 保存检查点
        if self.global_step % self.config.checkpoint_interval == 0:
            self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        return loss_dict
    
    def _log_training(self, loss_dict: Dict[str, float]):
        """记录训练日志到 TensorBoard"""
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
        
        # 记录学习率
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/lr', lr, self.global_step)
    
    def update_teacher_from_student(self):
        """
        将 Student 的权重复制到 Teacher
        （当 Student 表现超过 Teacher 时调用）
        """
        self.teacher_model.load_state_dict(self.student_model.state_dict())
        self._freeze_teacher()
        print(f"[Trainer] Teacher updated from Student at step {self.global_step}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'global_step': self.global_step,
            'games_trained': self.games_trained,
            'student_model_state_dict': self.student_model.state_dict(),
            'teacher_model_state_dict': self.teacher_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config_batch_size': self.config.batch_size,
            'config_learning_rate': self.config.learning_rate,
            'config_alpha': self.config.alpha,
            'config_beta': self.config.beta,
            'config_gamma': self.config.gamma,
        }
        torch.save(checkpoint, path)
        print(f"[Trainer] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.global_step = checkpoint['global_step']
        self.games_trained = checkpoint.get('games_trained', 0)
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', self.scheduler.state_dict()))
        
        # 更新配置（如果存在）
        if 'config_batch_size' in checkpoint:
            self.config.batch_size = checkpoint['config_batch_size']
        if 'config_learning_rate' in checkpoint:
            self.config.learning_rate = checkpoint['config_learning_rate']
        if 'config_alpha' in checkpoint:
            self.config.alpha = checkpoint['config_alpha']
        if 'config_beta' in checkpoint:
            self.config.beta = checkpoint['config_beta']
        if 'config_gamma' in checkpoint:
            self.config.gamma = checkpoint['config_gamma']
        
        print(f"[Trainer] Checkpoint loaded from {path}")
    
    async def train_loop(self, n_games: int = 100):
        """
        主训练循环
        
        Args:
            n_games: 训练游戏数量
        """
        print(f"[Trainer] Starting training loop for {n_games} games...")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Config: batch_size={self.config.batch_size}, "
              f"alpha={self.config.alpha}, beta={self.config.beta}, gamma={self.config.gamma}")
        
        start_time = time.time()
        
        for game_idx in range(n_games):
            game_id = f"train_game_{game_idx}"
            round_id = game_idx // 10
            
            # 自博弈生成数据
            trajectory, result, move_history = await self.self_play_game(game_id, round_id)
            
            # 将轨迹添加到 Replay Buffer
            if trajectory:
                self.replay_buffer.push(trajectory)
            
            self.games_trained += 1
            
            # 训练步骤
            if len(self.replay_buffer) >= self.config.batch_size:
                for _ in range(self.config.training_steps_per_game):
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    if batch:
                        loss_dict = self.train_step(batch)
                        
                        if game_idx % self.config.log_interval == 0:
                            print(f"Game {game_idx}: total_loss={loss_dict['total']:.4f}, "
                                  f"distill={loss_dict['distill']:.4f}, "
                                  f"selfplay={loss_dict['selfplay']:.4f}")
            
            # 定期更新 Teacher
            if game_idx > 0 and game_idx % 50 == 0:
                self.update_teacher_from_student()
        
        elapsed_time = time.time() - start_time
        print(f"[Trainer] Training completed: {n_games} games in {elapsed_time:.2f}s")
        print(f"[Trainer] Average: {elapsed_time / n_games:.2f}s per game")
        
        # 保存最终检查点
        self.save_checkpoint("checkpoint_final.pt")
        
        self.writer.close()


async def demo_training():
    """演示训练流程"""
    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        alpha=1.0,
        beta=0.5,
        gamma=0.01,
        replay_buffer_size=100,
        mcts_iterations=20,
        training_steps_per_game=5,
        checkpoint_interval=50,
        log_interval=10
    )
    
    trainer = Trainer(config=config)
    
    # 运行少量游戏进行演示
    await trainer.train_loop(n_games=20)
    
    print("\n=== Training Demo Complete ===")
    print(f"Global steps: {trainer.global_step}")
    print(f"Games trained: {trainer.games_trained}")
    print(f"Buffer size: {len(trainer.replay_buffer)} steps")


if __name__ == '__main__':
    asyncio.run(demo_training())
