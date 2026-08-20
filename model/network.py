"""
AlphaZero ResNet 模型架构

Input:  [B, 16, 8, 8]   # 16 通道：白方 6 子 + 黑方 6 子 + 当前回合 + 合法走法 mask
Output: policy [B, 4608], value [B, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """卷积块：Conv + BatchNorm + ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
  残差块：BN → ReLU → Conv → BN → ReLU → Conv + Shortcut
    """
    
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # 残差连接
        out += identity
        
        return out


class PolicyHead(nn.Module):
    """
  策略头：输出 4608 维策略向量
    
    4608 = 64 格 × 72 种走法
    - 56: queen-like moves (8 directions × 7 distances)
    - 8: knight moves
    - 8: underpromotions
    """
    
    def __init__(self, in_channels: int = 128, policy_channels: int = 32):
        super().__init__()
        self.policy_channels = policy_channels
        self.conv = nn.Conv2d(in_channels, policy_channels, 1)
        self.bn = nn.BatchNorm2d(policy_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(policy_channels * 64, 4608)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(-1, self.policy_channels * 64)
        out = self.fc(out)
        return out


class ValueHead(nn.Module):
    """
  价值头：输出标量价值 [-1, 1]
    1 = 胜，-1 = 负，0 = 和
    """
    
    def __init__(self, in_channels: int = 128, value_channels: int = 64):
        super().__init__()
        self.value_channels = value_channels
        self.conv = nn.Conv2d(in_channels, value_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(value_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(value_channels * 64, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(-1, self.value_channels * 64)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out


class AlphaZeroResNet(nn.Module):
    """
    AlphaZero 风格的 ResNet 模型
    
    架构：
    - 初始卷积层：16 → 128 通道
    - 8 个残差块
    - 策略头和价值头并行输出
    
    参数量：约 4.5-5.5M
    """
    
    def __init__(self, num_res_blocks: int = 8, channels: int = 128,
                 policy_channels: int = 32, value_channels: int = 64):
        """
        初始化模型
        
        Args:
            num_res_blocks: 残差块数量
            channels: 中间层通道数
            policy_channels: 策略头通道数
            value_channels: 价值头通道数
        """
        super().__init__()
        
        # 初始卷积层
        self.input_conv = ConvBlock(16, channels, kernel_size=3)
        
        # 残差塔
        self.residual_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
        
        # 策略头
        self.policy_head = PolicyHead(channels, policy_channels)
        
        # 价值头
        self.value_head = ValueHead(channels, value_channels)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, 16, 8, 8] 输入棋盘状态
            
        Returns:
            policy_logits: [B, 4608] 策略 logits
            value: [B, 1] 价值估计
        """
        # 初始卷积
        out = self.input_conv(x)
        
        # 残差块
        out = self.residual_tower(out)
        
        # 双头输出
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        
        return policy_logits, value
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """获取策略概率分布（带 softmax）"""
        policy_logits, _ = self.forward(x)
        return F.softmax(policy_logits, dim=-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """获取价值估计"""
        _, value = self.forward(x)
        return value


def create_model(num_res_blocks: int = 8, channels: int = 128,
                 pretrained_path: str = None) -> AlphaZeroResNet:
    """
    创建模型实例
    
    Args:
        num_res_blocks: 残差块数量
        channels: 中间层通道数
        pretrained_path: 预训练权重路径（可选）
        
    Returns:
        模型实例
    """
    model = AlphaZeroResNet(num_res_blocks=num_res_blocks, channels=channels)
    
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = AlphaZeroResNet()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, 16, 8, 8)
    
    with torch.no_grad():
        policy, value = model(dummy_input)
    
    print(f"Policy output shape: {policy.shape}")  # [B, 4608]
    print(f"Value output shape: {value.shape}")    # [B, 1]
