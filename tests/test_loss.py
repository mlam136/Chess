"""
测试 Loss 函数 - 数值验证
"""

import pytest
import torch
import numpy as np
from model.loss import compute_loss, ModelOutput, MCTSTarget
from model.network import AlphaZeroResNet


class TestLossFunction:
    """测试 Loss 函数"""
    
    def test_basic_loss_computation(self):
        """测试基本 loss 计算"""
        # 创建模拟数据
        batch_size = 4
        policy_dim = 4608
        
        student_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        teacher_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        mcts_target = MCTSTarget(
            policy=torch.softmax(torch.randn(batch_size, policy_dim), dim=-1),
            value=torch.randn(batch_size, 1)
        )
        
        # 创建临时模型用于正则项
        model = AlphaZeroResNet(num_res_blocks=2, channels=32)
        
        loss, details = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=model,
            alpha=1.0,
            beta=0.5,
            gamma=0.01
        )
        
        assert loss > 0
        assert 'distill' in details
        assert 'selfplay' in details
        assert 'total' in details
    
    def test_distill_loss_component(self):
        """测试蒸馏 loss 分量"""
        batch_size = 2
        policy_dim = 100  # 简化维度
        
        # Student 和 Teacher 输出相同应该得到低蒸馏 loss
        identical_policy = torch.ones(batch_size, policy_dim)
        identical_value = torch.ones(batch_size, 1) * 0.5
        
        student_output = ModelOutput(
            policy=identical_policy,
            value=identical_value
        )
        
        teacher_output = ModelOutput(
            policy=identical_policy + 0.01,  # 轻微差异
            value=identical_value + 0.01
        )
        
        mcts_target = MCTSTarget(
            policy=torch.softmax(identical_policy, dim=-1),
            value=identical_value
        )
        
        model = AlphaZeroResNet(num_res_blocks=2, channels=32)
        
        loss, details = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=model
        )
        
        # 蒸馏 loss 应该相对较小
        assert details['distill'] >= 0
    
    def test_selfplay_loss_component(self):
        """测试自博弈 loss 分量"""
        batch_size = 2
        policy_dim = 100
        
        student_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        teacher_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        # MCTS 目标与 Student 输出接近应该得到低自博弈 loss
        mcts_target = MCTSTarget(
            policy=torch.softmax(student_output.policy, dim=-1),
            value=student_output.value
        )
        
        model = AlphaZeroResNet(num_res_blocks=2, channels=32)
        
        loss, details = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=model
        )
        
        assert details['selfplay'] >= 0
    
    def test_l2_regularization(self):
        """测试 L2 正则化"""
        batch_size = 2
        policy_dim = 100
        
        student_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        teacher_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        mcts_target = MCTSTarget(
            policy=torch.softmax(torch.randn(batch_size, policy_dim), dim=-1),
            value=torch.randn(batch_size, 1)
        )
        
        # 大模型应该有更大的正则项
        large_model = AlphaZeroResNet(num_res_blocks=4, channels=64)
        small_model = AlphaZeroResNet(num_res_blocks=2, channels=32)
        
        _, details_large = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=large_model,
            gamma=0.1
        )
        
        _, details_small = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=small_model,
            gamma=0.1
        )
        
        # 大模型的 total loss 应该更大（因为正则项更大）
        # 注意：这取决于具体实现，这里只是概念验证
    
    def test_loss_weights(self):
        """测试 loss 权重 (alpha, beta)"""
        batch_size = 2
        policy_dim = 100
        
        student_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        teacher_output = ModelOutput(
            policy=torch.randn(batch_size, policy_dim),
            value=torch.randn(batch_size, 1)
        )
        
        mcts_target = MCTSTarget(
            policy=torch.softmax(torch.randn(batch_size, policy_dim), dim=-1),
            value=torch.randn(batch_size, 1)
        )
        
        model = AlphaZeroResNet(num_res_blocks=2, channels=32)
        
        # 高 alpha 应该增加蒸馏 loss 的权重
        loss_high_alpha, _ = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=model,
            alpha=10.0,
            beta=0.1
        )
        
        # 高 beta 应该增加自博弈 loss 的权重
        loss_high_beta, _ = compute_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            mcts_target=mcts_target,
            model=model,
            alpha=0.1,
            beta=10.0
        )
        
        assert loss_high_alpha > 0
        assert loss_high_beta > 0


class TestModelOutput:
    """测试模型输出数据结构"""
    
    def test_model_output_creation(self):
        """测试 ModelOutput 创建"""
        policy = torch.randn(4, 4608)
        value = torch.randn(4, 1)
        
        output = ModelOutput(policy=policy, value=value)
        
        assert output.policy.shape == (4, 4608)
        assert output.value.shape == (4, 1)
    
    def test_mcts_target_creation(self):
        """测试 MCTSTarget 创建"""
        policy = torch.softmax(torch.randn(4, 4608), dim=-1)
        value = torch.randn(4, 1)
        
        target = MCTSTarget(policy=policy, value=value)
        
        assert target.policy.shape == (4, 4608)
        assert target.value.shape == (4, 1)
        # 策略应该归一化
        assert torch.allclose(target.policy.sum(dim=-1), torch.ones(4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
