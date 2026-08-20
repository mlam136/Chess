📋 最终版项目指令（可直接交付执行）
项目：ChessRL — 多智能体国际象棋自博弈学习平台
0. 项目目标
构建一个本地可视化的实时国际象棋平台，核心机制为 Multi-Agent Self-Play + Teacher-Student Knowledge Distillation + AlphaZero MCTS。每个 AI 模型是一个"棋手"，通过反复博弈、蒸馏、身份轮换来逐步提升棋力。平台兼具教学演示和训练实验双重功能。

1. 全局超参数（运行时可热更新）
参数	含义	初始值	约束
N_AGENTS	初始 AI 棋手数量	8	≥ 4，偶数
WINDOW_X	计分滑动窗口（最近 X 局）	10	≥ 3
START_Y	身份分配起始局（第 Y 局后生效）	5	≥ 2
TIMEOUT_T	每步思考超时	120 s	> 0
MCTS_ITERATIONS	MCTS 每步搜索迭代次数	50	≥ 10
REPLAY_BUFFER_SIZE	回放缓冲区容量（局数）	500	≥ 50
BATCH_SIZE	蒸馏/自博弈训练 batch size	64	—
LR	学习率	1e-3	—
ALPHA	蒸馏 loss 权重	1.0	[0, ∞)
BETA	自博弈 loss 权重	0.5	[0, ∞)
GAMMA	L2 正则权重	0.01	[0, ∞)
CONCURRENT_GAMES	最大并发对局数	4	≤ N_AGENTS // 2
DISTILL_RATIO	Teacher:Student 比例	50 : 50	固定
所有参数存于 config/default.yaml，运行时通过 config.reload() 热更新，无需重启。

2. 核心博弈 & 训练循环
┌─────────────────────────────────────────────────────────┐
│                    MAIN LOOP                            │
│                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 1.MATCH   │───▶│ 2.PLAY      │───▶│ 3.SCORE      │  │
│  │ 随机配对   │    │ 并发执行对局  │    │ 滑动窗口计分  │  │
│  └───────────┘    └──────────────┘    └──────┬───────┘  │
│       ▲                                      │          │
│       │                                      ▼          │
│  ┌────┴─────────────────────────────────────────────┐   │
│  │ 4.IDENTITY  前50%→Teacher(冻结) / 后50%→Student   │   │
│  └────────────────────────┬─────────────────────────┘   │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 5.TRAIN     随机 1T↔1S 配对                          ││
│  │   · Student 用 MCTS 对 Teacher(冻结) 自博弈          ││
│  │   · 每步 (s, a, π_mcts, z) 写入 Replay Buffer        ││
│  │   · Loss = α·KL(π_T||π_S) + β·L_AZ + γ·||θ||²       ││
│  │   · 梯度只更新 Student                               ││
│  └─────────────────────────────────────────────────────┘│
│                           │                             │
│                           ▼ 回到 1 (下一轮)              │
└─────────────────────────────────────────────────────────┘
关键规则：

Teacher 参数在训练阶段完全冻结（requires_grad=False），仅作为前向推理的对手/参考。
每局结束后重新排序，身份可以每局翻转（从第 Y 局起生效）。
一个 Teacher 在一轮中可以同时被多个 Student 使用（推理无梯度，无冲突）。
3. 模型架构（推荐：ResNet-AlphaZero）
理由：与 MCTS 天然配套、社区实现成熟、8×8 输入适配良好、参数量适中（~5M），后续可替换为 Transformer 或加深。

Input:  [B, 16, 8, 8]   # 16通道: 白王/车/马/象/后/兵 + 黑方同 + 当前方 + 合法mask
  │
  ▼
Conv2d(16, 128, 3, pad=1) → ReLU → BatchNorm
  │
  ▼
ResBlock × 8  # 每个: BN→ReLU→Conv3×3→BN→ReLU→Conv3×3 + Shortcut
  │
  ├──────────────────────┐
  ▼                      ▼
# Policy Head          # Value Head
Conv2d(128,32,1)       Conv2d(128,64,3)
BN→ReLU→Flatten        BN→ReLU→Flatten
Linear(32*64, 4672)    Linear(64*8*8, 1) → Tanh
→ π: [B, 4672]        → v: [B, 1]  # (1=胜, 0=负, 0.5=平)
策略头：4672 维（64格 × 72种走法 = 4608，+64 个"无合法走法"标记，或直接用 4608）
价值头：标量 ∈ [−1, 1]
参数量：约 4.5 – 5.5M（8 个 ResBlock）
后续升级路径：加深 ResBlock 数 / 换 Transformer encoder / 加 attention
4. MCTS 搜索（AlphaZero 风格）
def mcts_search(model, board_state, iterations=50):
    """
    返回:
      π_mcts: [4608]  # 根节点走法概率（用于蒸馏 soft target）
      v_root:  float  # 根节点价值估计
    """
    root = MCTSNode(board_state)
    for _ in range(iterations):
        node = root
        # 1) Selection (PUCT)
        while node.is_terminal or node.is_fully_expanded:
            node = select_child(node, model)
        # 2) Expansion
        if not node.is_terminal:
            expand(node, model)
        # 3) Backpropagation
        backpropagate(node)
    # 输出根节点 π 分布
    π = get_root_policy(root)
    return π, root.value
使用 PUCT 公式 选择：Q(s,a) + λ · P(s,a) · √N(s) / (1 + N(s,a))
λ = 1.5（可调）
每步 MCTS 结果 (s, a, π_mcts, z) 写入 Replay Buffer
推理时 iterations 可动态调整（训练时 50，演示时 200+）
5. Replay Buffer 设计
class ReplayBuffer:
    """
    存储单元: (board_state_tensor, action_index, π_mcts_vector, reward_z, game_id, round_id)
    - 容量: REPLAY_BUFFER_SIZE (初始 500 局)
    - 采样: 按 game_id 均匀采样（避免同局相关性）
    - 淘汰: FIFO，最旧先出
    """
    def push(self, trajectory: list[StepRecord]): ...
    def sample(self, batch_size: int) -> Batch: ...
    def __len__(self): ...
6. Loss Function（最终版，无 GAN）
def compute_loss(student_output, teacher_output, mcts_target):
    """
    student_output:  (π_S: [B,4608], v_S: [B,1])
    teacher_output:  (π_T: [B,4608], v_T: [B,1])   # 冻结，无梯度
    mcts_target:     (π_mcts: [B,4608], z: [B,1])  # MCTS 搜索结果
    """
    # --- ① 蒸馏 loss（Student 对齐 Teacher）---
    L_policy_distill = F.kl_div(
        F.log_softmax(π_S, dim=-1),
        F.softmax(π_T, dim=-1),  # soft target
        reduction='batchmean'
    )
    L_value_distill = F.mse_loss(v_S, v_T)
    L_distill = L_policy_distill + L_value_distill

    # --- ② AlphaZero 自博弈 loss ---
    L_az_policy = -(mcts_target['π'] * F.log_softmax(π_S, dim=-1)).sum(-1).mean()
    L_az_value  = F.mse_loss(v_S, mcts_target['z'])
    L_selfplay  = L_az_policy + L_az_value

    # --- ③ 正则 ---
    L_reg = sum(p.pow(2).sum() for p in student.parameters()) * GAMMA

    # --- 总 loss ---
    L_total = ALPHA * L_distill + BETA * L_selfplay + L_reg
    return L_total, {
        'distill': L_distill.item(),
        'selfplay': L_selfplay.item(),
        'total': L_total.item()
    }
训练时：Student 先对 Teacher 走一局（MCTS 搜索），同时用 Replay Buffer 中的自博弈数据做 batch 训练。两条 loss 路径并行。

7. 棋盘状态编码（模型输入 & 可视化分离）
def encode_board(board: chess.Board) -> torch.Tensor:
    """
    返回: [16, 8, 8] 张量
    通道 0-5:  白方 (K, Q, R, B, N, P)  one-hot 平面
    通道 6-11: 黑方 (K, Q, R, B, N, P)  one-hot 平面
    通道 12:   当前回合 (白=1, 黑=0) 全1/全0
    通道 13:   合法走法目标格 mask
    通道 14:   合法走法源格 mask
    通道 15:   可吃过路兵 / 可王车易位标记 (简化)
    """
图片资源 (./resource/png/) 仅用于 Pygame 渲染层，与模型输入完全解耦。

8. 走法校验 & 违规处理
class MoveValidator:
    def validate(self, agent_id: str, proposed_move: str, board: chess.Board) -> MoveResult:
        """
        检查:
          1. 格式合法 (SAN/UCI)
          2. 是否在 board.legal_moves 中
          3. 是否超时 (> TIMEOUT_T)
          4. 是否重复走法 (三次重复 → 和局)
          5. 是否 50 步无吃子/兵移动 → 和局
          6. 是否将死/逼和 → 终局判定
        返回: MoveResult(status, penalty, reason)
        """
9. 可视化（Pygame，单窗口多盘）
┌─────────────────────────────────────────────────────────────┐
│  ChessRL v1.0                    [暂停] [加速] [设置]        │
├────────────────────┬────────────────────────────────────────┤
│  Game 1: A vs B    │  Game 3: C vs D                        │
│  ┌──────────────┐  │  ┌──────────────┐                      │
│  │  8×8 Board   │  │  │  8×8 Board   │                      │
│  │  (pieces)    │  │  │  (pieces)    │                      │
│  └──────────────┘  │  └──────────────┘                      │
│  Score: +1 / -0    │  Score: 0 / +1                         │
├────────────────────┼────────────────────────────────────────┤
│  Game 2: E vs F    │  [Scoreboard]  [Loss Curve]  [Log]     │
│  ┌──────────────┐  │  ┌─────────────────────────────────┐   │
│  │  8×8 Board   │  │  │  Agent  │ Score │ Role │ Win%   │   │
│  └──────────────┘  │  │  A      │  +2   │ T    │  75%   │   │
│  Score: 0 / 0      │  │  B      │  -1   │ S    │  50%   │   │
│                    │  │  C      │  +1   │ T    │  60%   │   │
│                    │  │  D      │  -2   │ S    │  40%   │   │
│                    │  └─────────────────────────────────┘   │
└────────────────────┴────────────────────────────────────────┘
布局：网格排列，每盘占一个 cell（默认 2×2 = 4 盘），可拖拽调整
交互：点击棋盘选格 → 点击目标格走子（真人模式）；模型模式自动播放
动画：走子平滑移动、吃子淡出、将军闪烁
侧栏：实时 scoreboard + loss 曲线（pygame 自绘或内嵌 matplotlib widget）
资源：从 ./resource/png/ 加载棋盘底图和棋子精灵图
10. 并发架构
┌──────────────────────────────────────────────┐
│              asyncio Event Loop              │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │ Game 1  │ │ Game 2  │ │ Game M  │  ← N 个并发对局
│  │ (async) │ │ (async) │ │ (async) │
│  └────┬────┘ └────┬────┘ └────┬────┘         │
│       │            │           │             │
│       ▼            ▼           ▼             │
│  ┌─────────────────────────────────────────┐ │
│  │     Model Inference Pool (GPU)          │ │
│  │  (batched forward, MCTS 迭代)            │ │
│  └─────────────────────────────────────────┘ │
│       │                                      │
│       ▼                                      │
│  ┌─────────────────────────────────────────┐ │
│  │     Event Bus (走法/胜负/身份变更)        │ │
│  └──────────────────┬──────────────────────┘ │
│                     ▼                        │
│  ┌─────────────────────────────────────────┐ │
│  │     Pygame Renderer (主线程)             │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
对局逻辑：asyncio 协程（每步 MCTS 推理 await，不阻塞）
模型推理：PyTorch 半精度 + batch（多盘同时推理合并为一次 forward）
渲染：Pygame 主线程，从 Event Bus 拉取最新帧状态
训练：独立 asyncio 任务，在 Student 空闲时执行
11. 最终目录结构
chess-rl/
│
├── config/
│   ├── default.yaml              # 所有超参（N, X, Y, T, MCTS_iter, α, β, γ, ...）
│   └── hot_reload.py             # 运行时热更新
│
├── engine/
│   ├── __init__.py
│   ├── board.py                  # 棋盘状态封装 (python-chess wrapper)
│   ├── game.py                   # 单盘对局生命周期 (setup → play → end)
│   ├── rules.py                  # 走法合法性、超时、和局、终局判定
│   ├── scheduler.py              # 匹配、并发调度、角色分配
│   └── scoring.py                # 滑动窗口计分、身份排序
│
├── mcts/
│   ├── __init__.py
│   ├── node.py                   # MCTSNode (Q, N, P, prior, children)
│   ├── search.py                 # PUCT 选择、扩展、回溯
│   └── policy.py                 # 根节点 π 提取
│
├── model/
│   ├── __init__.py
│   ├── network.py                # ResNet-AlphaZero (policy + value head)
│   ├── encoder.py                # encode_board() → [16,8,8] Tensor
│   ├── loss.py                   # compute_loss() 蒸馏 + AZ + 正则
│   ├── trainer.py                # 训练循环 (蒸馏 + Replay Buffer 采样)
│   ├── replay_buffer.py          # ReplayBuffer (FIFO, 按局采样)
│   └── checkpoint.py             # save/load/版本管理
│
├── agent/
│   ├── __init__.py
│   ├── base.py                   # Agent 抽象基类 (think(board) → move)
│   ├── model_agent.py            # MCTS + Network 推理
│   ├── teacher_agent.py          # 冻结 Teacher (推理 only, 不更新)
│   └── human_agent.py            # 真人 (读取 UI 输入)
│
├── viz/
│   ├── __init__.py
│   ├── app.py                    # Pygame 主循环 (单窗口, 多盘网格)
│   ├── board_widget.py           # 单盘渲染 (棋盘+棋子+动画)
│   ├── scoreboard.py             # 分数/身份面板
│   ├── loss_chart.py             # 实时 loss 曲线
│   ├── log_panel.py              # 文字日志
│   └── assets.py                 # 加载 ./resource/png/ 精灵图
│
├── api/
│   ├── __init__.py
│   ├── events.py                 # 事件总线 (EventEmitter)
│   ├── board_state.py            # get_board_state() 实时输出
│   └── move_validator.py         # 走法校验 (格式/合法性/超时/违规)
│
├── tests/
│   ├── __init__.py
│   ├── test_rules.py             # 走法合法性、和局、终局
│   ├── test_scoring.py           # 计分、身份分配边界
│   ├── test_mcts.py              # MCTS 搜索正确性
│   ├── test_loss.py              # loss 数值验证
│   └── test_concurrency.py       # 多盘并发不冲突
│
├── resource/
│   └── png/                      # [已有] 棋子 & 棋盘图片
│
├── scripts/
│   ├── train.py                  # 纯训练 (无UI, 跑 N 轮)
│   ├── eval.py                   # 评估 (模型 vs 模型 / 模型 vs 引擎)
│   └── demo.py                   # 演示模式 (可视化 + 自动对局)
│
├── main.py                       # CLI 入口: python main.py --mode {play|train|demo}
├── requirements.txt
└── README.md
12. requirements.txt
python-chess>=1.0
torch>=2.1
torchaudio>=2.1        # 可选，若加音频
pygame>=2.5
numpy>=1.24
pydantic>=2.0
pyyaml>=6.0
tqdm>=4.65
tensorboard>=2.14
pytest>=7.4
13. 分阶段实施计划
阶段	交付物	验收标准	预估
P0 — 引擎 & 可视化	engine/ + viz/ + resource/	两个 random agent 在 Pygame 窗口中完成一局；走子动画流畅；非法走法被拦截	2 天
P1 — 并发 & 计分	scheduler.py + scoring.py + api/	4 盘同时对局；计分正确；身份分配正确；日志完整	2 天
P2 — 模型 & MCTS	model/network.py + mcts/ + agent/model_agent.py	模型能合法走完一局；MCTS 搜索正确（与已知局面验证）	3 天
P3 — 训练循环	trainer.py + loss.py + replay_buffer.py	8 个 agent 跑 10 轮完整循环；loss 下降；Teacher 参数冻结确认	3 天
P4 — 监控 & 调优	viz/loss_chart.py + tensorboard + 消融	实时 loss 曲线；scoreboard 更新；α/β 消融对比	2 天
P5 — 真人 & 打磨	agent/human_agent.py + UI 交互	真人可点击走子；与模型对弈；设置面板热更新参数	2 天
P6 — 测试 & 文档	tests/ + README.md	pytest 全绿；README 含架构图、快速开始、FAQ	1 天
总计：约 15 个工作日（单人）

14. 后续可扩展方向（P7+）
模型架构升级：Transformer encoder / 更深 ResNet / 多尺度
搜索增强：PVS / 变深度 MCTS / 在线推理
多策略蒸馏：Top-K Teacher 加权蒸馏（而非单一 Teacher）
课程学习：从 4×4 棋盘 → 8×8
引擎对标：与 Stockfish (depth 5-15) 对弈评估 Elo
Web 可视化：Flask + WebSocket 推送，手机可看
