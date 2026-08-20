"""
Loss 函数 - 蒸馏 + AlphaZero 自博弈 + 正则

Loss = α·L_distill + β·L_selfplay + γ·L_reg

其中：
- L_distill = KL(π_S || π_T) + MSE(v_S, v_T)
- L_selfplay = -Σ(π_mcts · log(π_S)) + MSE(v_S, z)
- L_reg = Σ||θ||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """模型输出封装"""
    policy_logits: torch.Tensor  # [B, 4608]
    value: torch.Tensor          # [B, 1]


@dataclass
class MCTSTarget:
    """MCTS 目标封装"""
    policy: torch.Tensor  # [B, 4608] MCTS 策略分布
    value: torch.Tensor   # [B, 1] 游戏结果 (1=胜，-1=负，0=和)


def compute_loss(student_output: ModelOutput, 
                 teacher_output: Optional[ModelOutput],
                 mcts_target: MCTSTarget,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.01,
                 student_model: nn.Module = None) -> Tuple[torch.Tensor, Dict]:
    """
    计算总 loss
    
    Args:
        student_output: Student 模型输出
        teacher_output: Teacher 模型输出（可选，用于蒸馏）
        mcts_target: MCTS 搜索结果
        alpha: 蒸馏 loss 权重
        beta: 自博弈 loss 权重
        gamma: L2 正则权重
        student_model: Student 模型（用于计算 L2 正则）
        
    Returns:
        total_loss: 总 loss
        loss_dict: 各分项 loss 字典
    """
    loss_dict = {}
    
    # --- ① 蒸馏 loss（Student 对齐 Teacher）---
    if teacher_output is not None:
        l_policy_distill = F.kl_div(
            F.log_softmax(student_output.policy_logits, dim=-1),
            F.softmax(teacher_output.policy_logits, dim=-1),
            reduction='batchmean'
        )
        l_value_distill = F.mse_loss(student_output.value, teacher_output.value)
        l_distill = l_policy_distill + l_value_distill
    else:
        l_distill = torch.tensor(0.0, device=student_output.policy_logits.device)
    
    loss_dict['distill'] = l_distill.item()
    
    # --- ② AlphaZero 自博弈 loss ---
    # 策略 loss: 交叉熵
    l_az_policy = -(mcts_target.policy * F.log_softmax(student_output.policy_logits, dim=-1)).sum(-1).mean()
    
    # 价值 loss: MSE
    l_az_value = F.mse_loss(student_output.value, mcts_target.value)
    
    l_selfplay = l_az_policy + l_az_value
    loss_dict['selfplay'] = l_selfplay.item()
    
    # --- ③ L2 正则 ---
    if student_model is not None and gamma > 0:
        l_reg = sum(p.pow(2).sum() for p in student_model.parameters()) * gamma
    else:
        l_reg = torch.tensor(0.0, device=student_output.policy_logits.device)
    
    loss_dict['regularization'] = l_reg.item()
    
    # --- 总 loss ---
    total_loss = alpha * l_distill + beta * l_selfplay + l_reg
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    计算 KL 散度 D_KL(P || Q)
    
    Args:
        p: [B, N] 概率分布 P
        q: [B, N] 概率分布 Q
        
    Returns:
        KL 散度标量
    """
    p_log_probs = F.log_softmax(p, dim=-1)
    q_probs = F.softmax(q, dim=-1)
    return F.kl_div(p_log_probs, q_probs, reduction='batchmean')


def compute_cross_entropy(policy_logits: torch.Tensor, 
                          target_policy: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失
    
    Args:
        policy_logits: [B, 4608] 策略 logits
        target_policy: [B, 4608] 目标策略分布
        
    Returns:
        交叉熵损失标量
    """
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -(target_policy * log_probs).sum(-1).mean()


if __name__ == '__main__':
    # 测试 loss 计算
    batch_size = 4
    
    # 模拟数据
    student_logits = torch.randn(batch_size, 4608)
    teacher_logits = torch.randn(batch_size, 4608)
    mcts_policy = F.softmax(torch.randn(batch_size, 4608), dim=-1)
    mcts_value = torch.randn(batch_size, 1)
    
    student_out = ModelOutput(policy_logits=student_logits, value=torch.randn(batch_size, 1))
    teacher_out = ModelOutput(policy_logits=teacher_logits, value=torch.randn(batch_size, 1))
    mcts_target = MCTSTarget(policy=mcts_policy, value=mcts_value)
    
    # 创建简单模型用于正则项
    dummy_model = nn.Linear(10, 10)
    
    total_loss, losses = compute_loss(
        student_out, teacher_out, mcts_target,
        alpha=1.0, beta=0.5, gamma=0.01,
        student_model=dummy_model
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Distill loss: {losses['distill']:.4f}")
    print(f"Selfplay loss: {losses['selfplay']:.4f}")
    print(f"Regularization: {losses['regularization']:.6f}")
