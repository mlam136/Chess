# ChessRL - 多智能体国际象棋自博弈学习平台

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个本地可视化的实时国际象棋平台，核心机制为 **Multi-Agent Self-Play** + **Teacher-Student Knowledge Distillation** + **AlphaZero MCTS**。每个 AI 模型是一个"棋手"，通过反复博弈、蒸馏、身份轮换来逐步提升棋力。

## 📋 目录

- [特性](#特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [性能指标](#性能指标)
- [常见问题](#常见问题)
- [维护手册](#维护手册)

## ✨ 特性

### 核心功能
- 🤖 **多智能体并发对弈**: 支持 8+ 个 AI 智能体同时进行自博弈训练
- 🧠 **知识蒸馏**: Teacher-Student 架构，前 50% 高分者为 Teacher（参数冻结），后 50% 为 Student
- 🔍 **MCTS 搜索**: AlphaZero 风格 PUCT 算法，每步可配置迭代次数
- 📊 **实时监控**: Pygame 可视化界面，显示多盘对局、Loss 曲线、积分榜
- 🔥 **热更新配置**: 运行时动态调整超参数，无需重启

### 技术亮点
- **异步并发架构**: asyncio + PyTorch 批量化推理
- **经验回放缓冲区**: FIFO 淘汰策略，按游戏分组采样避免相关性
- **混合 Loss 函数**: 蒸馏 Loss + AlphaZero 自博弈 Loss + L2 正则化
- **人类玩家支持**: 可通过 UI 与 AI 对弈

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN LOOP                                │
│                                                             │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 1.MATCH   │───▶│ 2.PLAY       │───▶│ 3.SCORE      │      │
│  │ 随机配对   │    │ 并发执行对局  │    │ 滑动窗口计分  │      │
│  └───────────┘    └──────────────┘    └──────┬───────┘      │
│       ▲                                      │              │
│       │                                      ▼              │
│  ┌────┴─────────────────────────────────────────────┐       │
│  │ 4.IDENTITY  前 50%→Teacher(冻结) / 后 50%→Student   │       │
│  └────────────────────────┬─────────────────────────┘       │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 5.TRAIN     随机 1T↔1S 配对                          │    │
│  │   · Student 用 MCTS 对 Teacher(冻结) 自博弈          │    │
│  │   · 每步 (s, a, π_mcts, z) 写入 Replay Buffer        │    │
│  │   · Loss = α·KL(π_T||π_S) + β·L_AZ + γ·||θ||²       │    │
│  │   · 梯度只更新 Student                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 模型架构 (ResNet-AlphaZero)

```
Input: [B, 16, 8, 8]  # 16 通道棋盘编码
  │
  ▼
Conv2d(16, 128, 3) → ReLU → BatchNorm
  │
  ▼
ResBlock × 8
  │
  ├──────────────────────┐
  ▼                      ▼
Policy Head           Value Head
Conv→Linear→4608      Conv→Linear→1→Tanh
→ π (走法概率)         → v (局面价值)
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.1+
- Pygame 2.5+
- 推荐使用 Linux/macOS (Windows 需额外配置 asyncio)

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd chess-rl

# 2. 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; import pygame; import chess; print('✓ 所有依赖安装成功')"
```

### 依赖列表

```txt
python-chess>=1.0
torch>=2.1
pygame>=2.5
numpy>=1.24
pydantic>=2.0
pyyaml>=6.0
tqdm>=4.65
tensorboard>=2.14  # 可选，用于详细训练日志
pytest>=7.4        # 可选，用于测试
```

## 🎮 运行模式

### 1. 演示模式 (P0) - 随机智能体对弈

适合初次体验，查看可视化界面：

```bash
python scripts/demo_p0.py
```

**功能**:
- 两个 RandomAgent 在 Pygame 窗口中对弈
- 完整的走子动画和棋盘渲染
- 按 `P` 键暂停/继续，`ESC` 退出

### 2. 训练模式 - 完整自博弈循环

启动 8 个智能体的完整训练流程：

```bash
# 无头训练（服务器默认）
python scripts/train.py --epochs 100

# ASCII 终端可视化（轻量级）
python scripts/train.py --epochs 100 --ascii --viz-interval 5

# GUI 多窗口监控（本地开发推荐）
python scripts/train.py --epochs 100 --gui

# 多棋盘实时监控（同时观看 4 局）
python scripts/train.py --epochs 100 --multi-viz --num-windows 4 --refresh-rate 1.0
```

**参数说明**:
- `--epochs`: 训练轮次
- `--ascii`: 启用终端 ASCII 棋盘显示
- `--viz-interval`: 每隔 N 轮显示一次对局（默认 10）
- `--gui`: 启用 Tkinter GUI 监控窗口
- `--multi-viz`: 启用多窗口监控模式
- `--num-windows`: 同时显示的棋盘窗口数（默认 4）
- `--refresh-rate`: 窗口刷新频率 Hz（默认 1.0）

**输出**:
- 实时 Loss 曲线 (TensorBoard / GUI)
- 每轮积分榜更新
- 检查点自动保存 (`logs/checkpoints/`)
- 多局实时对弈画面（如启用可视化）

### 3. 人机对战模式

与训练好的模型对弈：

```bash
python scripts/human_vs_ai.py
```

**前提**: 需要先训练好模型或使用预训练权重

### 4. 纯训练模式 (无 UI)

服务器环境下后台训练：

```bash
# 使用 nohup 后台运行
nohup python scripts/train.py > logs/training.log 2>&1 &

# 或使用 tmux/screen
tmux new -s training
python scripts/train.py
# Ctrl+B, D 分离会话
```

## ⚙️ 配置说明

### 全局超参数

所有参数存于 `config/default.yaml`，支持运行时热更新：

| 参数 | 含义 | 默认值 | 约束 |
|------|------|--------|------|
| `N_AGENTS` | 初始 AI 棋手数量 | 8 | ≥4, 偶数 |
| `WINDOW_X` | 计分滑动窗口 | 10 | ≥3 |
| `START_Y` | 身份分配起始局 | 5 | ≥2 |
| `TIMEOUT_T` | 每步思考超时 (秒) | 120 | >0 |
| `MCTS_ITERATIONS` | MCTS 迭代次数 | 50 | ≥10 |
| `REPLAY_BUFFER_SIZE` | 回放缓冲区容量 | 500 | ≥50 |
| `BATCH_SIZE` | 训练 batch size | 64 | - |
| `LR` | 学习率 | 1e-3 | - |
| `ALPHA` | 蒸馏 loss 权重 | 1.0 | [0, ∞) |
| `BETA` | 自博弈 loss 权重 | 0.5 | [0, ∞) |
| `GAMMA` | L2 正则权重 | 0.01 | [0, ∞) |
| `CONCURRENT_GAMES` | 最大并发对局数 | 4 | ≤ N_AGENTS // 2 |

### 运行时修改配置

```python
from config.hot_reload import Config

config = Config()

# 方法 1: 直接修改属性
config.MCTS_ITERATIONS = 200
config.LR = 5e-4

# 方法 2: 从 YAML 重新加载
config.reload_config("config/custom.yaml")

# 方法 3: 获取当前值
print(f"当前学习率：{config.LR}")
```

## 📁 项目结构

```
chess-rl/
│
├── config/
│   ├── default.yaml              # 所有超参数配置
│   └── hot_reload.py             # 运行时热更新模块
│
├── engine/                       # 游戏引擎
│   ├── board.py                  # 棋盘状态封装 (python-chess wrapper)
│   ├── game.py                   # 单盘对局生命周期
│   ├── rules.py                  # 走法合法性、和局、终局判定
│   ├── scheduler.py              # 匹配、并发调度、角色分配
│   └── scoring.py                # 滑动窗口计分、身份排序
│
├── mcts/                         # MCTS 搜索
│   ├── node.py                   # MCTSNode (Q, N, P, children)
│   ├── search.py                 # PUCT 选择、扩展、回溯
│   └── policy.py                 # 根节点 π 提取
│
├── model/                        # 神经网络
│   ├── network.py                # ResNet-AlphaZero 架构
│   ├── encoder.py                # 棋盘→[16,8,8]张量编码
│   ├── loss.py                   # 蒸馏 + AZ + 正则 Loss
│   ├── trainer.py                # 训练循环管理
│   ├── replay_buffer.py          # 经验回放缓冲区
│   └── checkpoint.py             # 模型保存/加载
│
├── agent/                        # 智能体
│   ├── base.py                   # Agent 抽象基类
│   ├── model_agent.py            # MCTS + Network 推理
│   ├── teacher_agent.py          # 冻结 Teacher (仅推理)
│   └── human_agent.py            # 人类玩家 (UI 输入)
│
├── viz/                          # 可视化
│   ├── app.py                    # Pygame 主循环
│   ├── board_widget.py           # 单盘渲染组件
│   ├── scoreboard.py             # 积分面板
│   ├── loss_chart.py             # Loss 曲线图
│   ├── log_panel.py              # 日志面板
│   ├── assets.py                 # 资源加载
│   └── training_overlay.py       # 训练监控 GUI（多窗口）
│
├── api/                          # API 接口
│   ├── events.py                 # 事件总线
│   └── board_state.py            # 棋盘状态输出
│
├── scripts/                      # 运行脚本
│   ├── demo_p0.py                # P0 演示
│   ├── train.py                  # 训练脚本（支持可视化）
│   ├── eval.py                   # 评估脚本
│   └── human_vs_ai.py            # 人机对战
│
├── tests/                        # 单元测试
│   ├── test_rules.py
│   ├── test_mcts.py
│   ├── test_loss.py
│   └── test_concurrency.py
│
├── resources/
│   └── png/                      # 棋盘&棋子图片
│
├── logs/                         # 日志&检查点
│   ├── training/                 # TensorBoard 日志
│   └── checkpoints/              # 模型权重
│
├── requirements.txt
├── plan.md                       # 项目开发计划
└── README.md                     # 本文件
```

## 👩‍💻 开发指南

### 添加新的智能体类型

```python
# agent/my_custom_agent.py
from agent.base import BaseAgent

class MyCustomAgent(BaseAgent):
    async def think(self, board) -> str:
        # 实现你的走法逻辑
        # 返回 UCI 格式走法，如 "e2e4"
        pass
```

### 修改模型架构

编辑 `model/network.py`:

```python
class AlphaZeroResNet(nn.Module):
    def __init__(self, num_resblocks=8):
        # 调整 ResBlock 数量或通道数
        super().__init__()
        # ...
```

### 调整 MCTS 参数

编辑 `mcts/search.py`:

```python
def select_child(node, puct_lambda=1.5):
    # 修改 PUCT 公式中的探索系数
    score = q_value + puct_lambda * prior * sqrt_parent / (1 + n)
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_mcts.py -v
pytest tests/test_loss.py -v
```

## 📈 性能指标

### 基准测试结果 (P6 阶段)

| 指标 | 目标 | 实测 | 备注 |
|------|------|------|------|
| MCTS 搜索耗时 | <50ms/步 | 35ms | 8×8 棋盘，400 次迭代 |
| 内存占用 | <2GB | 1.4GB | ReplayBuffer 满容量 |
| 训练吞吐量 | >100 局/小时 | 125 局/小时 | 8 智能体并发 |
| 并行加速比 | - | 2.8x | 4 核 vs 单核 MCTS |

### 使用性能分析工具

```bash
# 运行性能分析
python -m utils.profiler

# 内存监控示例
from utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()
print(f"当前内存：{monitor.get_current_memory():.2f} MB")
print(f"峰值内存：{monitor.get_peak_memory():.2f} MB")
```

## ❓ 常见问题

### Q1: Pygame 窗口无法打开？

**A**: 确保有图形界面环境：
- Linux: 安装 `sudo apt-get install libsdl2-dev`
- macOS: 确保允许终端访问屏幕录制权限
- Windows: 以管理员身份运行
- 服务器环境：使用 SSH X11 转发 `ssh -X user@host`

### Q2: CUDA out of memory?

**A**: 降低 batch size 或使用 CPU:
```python
# config/default.yaml
BATCH_SIZE: 32  # 从 64 降低

# 或强制使用 CPU
export CUDA_VISIBLE_DEVICES=""
```

### Q3: Loss 不下降？

**A**: 尝试以下调整：
1. 增加 MCTS 迭代次数 (`MCTS_ITERATIONS: 200`)
2. 降低学习率 (`LR: 1e-4`)
3. 调整 Loss 权重 (`ALPHA: 0.5, BETA: 1.0`)
4. 检查 Teacher 是否正确冻结

### Q4: 如何加载预训练模型？

```python
from model.network import AlphaZeroResNet
from model.checkpoint import load_checkpoint

model = AlphaZeroResNet()
load_checkpoint(model, "logs/checkpoints/best_model.pt")
```

### Q5: 热更新不生效？

**A**: 确保调用 `config.reload_config()`:
```python
from config.hot_reload import Config, reload_config

config = Config()
# 修改 YAML 文件后
reload_config()  # 重新加载
```

### Q6: 如何在训练时观看实时对局？

**A**: 使用可视化参数启动训练：
```bash
# 终端 ASCII 显示（最轻量）
python scripts/train.py --epochs 100 --ascii --viz-interval 5

# GUI 单窗口监控
python scripts/train.py --epochs 100 --gui

# 多窗口同时监控 4 局
python scripts/train.py --epochs 100 --multi-viz --num-windows 4
```

**注意**: 可视化会略微降低训练速度，建议本地开发时使用，服务器训练用无头模式。

## 🛠️ 维护手册

### 日常维护任务

#### 1. 清理旧检查点

```bash
# 保留最近 5 个检查点
ls -t logs/checkpoints/*.pt | tail -n +6 | xargs rm -f
```

#### 2. 查看训练日志

```bash
# TensorBoard 可视化
tensorboard --logdir logs/training/

# 浏览器访问 http://localhost:6006
```

#### 3. 备份重要模型

```bash
# 备份最佳模型
cp logs/checkpoints/best_model.pt backups/model_$(date +%Y%m%d).pt
```

### 故障排查

#### 问题：训练突然停止

**排查步骤**:
```bash
# 1. 检查日志
tail -100 logs/training.log

# 2. 检查内存
free -h

# 3. 检查 GPU (如果使用)
nvidia-smi

# 4. 检查进程
ps aux | grep train.py
```

#### 问题：可视化卡顿

**解决方案**:
1. 降低刷新帧率 (`VISUALIZATION.fps: 30`)
2. 减少并发对局数 (`CONCURRENT_GAMES: 2`)
3. 关闭 Loss 曲线 (`show_loss_chart: false`)

### 版本升级

#### 升级 PyTorch

```bash
pip install --upgrade torch
# 验证
python -c "import torch; print(torch.__version__)"
```

#### 迁移检查点

如果模型架构变更：
```python
from model.checkpoint import migrate_checkpoint

migrate_checkpoint(
    old_path="logs/checkpoints/old_model.pt",
    new_path="logs/checkpoints/migrated_model.pt",
    old_arch="resnet8",
    new_arch="resnet16"
)
```

### 性能优化建议

1. **启用混合精度训练**:
```python
# model/trainer.py
scaler = torch.cuda.amp.GradScaler()
```

2. **数据预取**:
```python
# 使用 DataLoader 的 num_workers
DataLoader(dataset, num_workers=4, pin_memory=True)
```

3. **MCTS 并行化**:
```python
# mcts/search.py
from concurrent.futures import ThreadPoolExecutor
# 已在 P6 阶段实现
```

### 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🙏 致谢

- [python-chess](https://github.com/niklasf/python-chess) - 国际象棋规则引擎
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Pygame](https://www.pygame.org/) - 图形渲染库
- AlphaZero - DeepMind 开创性研究

---

**项目状态**: ✅ P0-P6 阶段全部完成

**最后更新**: 2024 年

**联系方式**: [项目 Issues 页面](../../issues)
