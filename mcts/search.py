"""
MCTS 搜索算法 - AlphaZero 风格 PUCT 搜索
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import chess

from engine.board import Board
from .node import MCTSNode


# 默认 PUCT 探索系数
DEFAULT_C_PUCT = 1.5


def mcts_search(board: Board, model=None, iterations: int = 50, 
                c_puct: float = DEFAULT_C_PUCT) -> Tuple[np.ndarray, float]:
    """
    MCTS 搜索主函数
    
    Args:
        board: 当前棋盘状态
        model: 神经网络模型（用于评估和策略先验）
               如果为 None，则使用随机 rollout
        iterations: 搜索迭代次数
        c_puct: PUCT 探索系数
        
    Returns:
        pi_mcts: [4608] 根节点走法概率分布
        v_root: 根节点价值估计
    """
    # 创建根节点
    root = MCTSNode(board=board)
    
    # 如果有模型，获取先验概率和价值
    if model is not None:
        priors, value = _get_model_prior(model, board)
        # 初始化根节点的子节点先验
        _expand_node(root, priors)
    
    for _ in range(iterations):
        node = root
        
        # 1) Selection - 选择直到叶子节点或未完全扩展节点
        while node.is_terminal or node.is_fully_expanded:
            if node.is_terminal:
                break
            node = select_child(node, model, c_puct)
        
        # 2) Expansion - 扩展未终止的叶子节点
        if not node.is_terminal and not node.is_fully_expanded:
            if model is not None:
                expand(node, model)
            else:
                # 无模型时使用随机 rollout
                _random_rollout(node)
        
        # 3) Backpropagation - 反向传播更新路径上的节点
        backpropagate(node)
    
    # 返回根节点的策略分布和价值
    pi_mcts = get_root_policy(root)
    v_root = root.value
    
    return pi_mcts, v_root


def select_child(node: MCTSNode, model=None, c_puct: float = DEFAULT_C_PUCT) -> MCTSNode:
    """
    使用 PUCT 公式选择子节点
    
    PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    
    Args:
        node: 当前节点
        model: 神经网络模型
        c_puct: 探索系数
        
    Returns:
        选中的子节点
    """
    best_score = -float('inf')
    best_child = None
    
    for move, child in node.children.items():
        # PUCT 公式
        exploitation = child.Q
        exploration = c_puct * child.P * np.sqrt(node.N) / (1 + child.N)
        score = exploitation + exploration
        
        if score > best_score:
            best_score = score
            best_child = child
    
    return best_child if best_child is not None else list(node.children.values())[0]


def expand(node: MCTSNode, model) -> None:
    """
    扩展节点 - 为所有未访问的合法走法创建子节点
    
    Args:
        node: 要扩展的节点
        model: 神经网络模型，用于获取先验概率
    """
    if node.is_terminal:
        return
    
    # 获取所有合法走法的先验概率
    priors, _ = _get_model_prior(model, node.board)
    
    for move in node.legal_moves:
        if move not in node.children:
            # 获取该走法的先验概率
            move_idx = _move_to_index(move)
            prior = priors[move_idx] if move_idx < len(priors) else 0.0
            
            # 创建子节点
            node.add_child(move, prior)


def _expand_node(node: MCTSNode, priors: np.ndarray) -> None:
    """
    使用给定的先验概率扩展节点
    
    Args:
        node: 要扩展的节点
        priors: [4608] 先验概率数组
    """
    if node.is_terminal:
        return
    
    for move in node.legal_moves:
        if move not in node.children:
            move_idx = _move_to_index(move)
            prior = priors[move_idx] if move_idx < len(priors) else 0.0
            node.add_child(move, prior)


def backpropagate(node: MCTSNode) -> None:
    """
    反向传播 - 从叶子节点向上传播价值估计
    
    Args:
        node: 叶子节点（已评估）
    """
    # 获取要传播的价值
    value = node.value
    
    current = node
    while current is not None:
        current.N += 1
        # 更新 Q 值：累积价值并取平均
        # 注意：对于黑方回合，需要反转价值符号
        if current.parent is not None:
            # 判断是否是对手回合
            is_opponent_turn = not current.parent.board.turn
            if is_opponent_turn:
                value = -value
        
        # 增量更新平均值
        current.Q += (value - current.Q) / current.N
        
        current = current.parent


def _random_rollout(node: MCTSNode) -> None:
    """
    随机 rollout 评估叶子节点（无模型时的回退策略）
    
    Args:
        node: 叶子节点
    """
    board = node.board.copy()
    
    # 随机模拟到游戏结束
    while not board.is_game_over:
        legal_moves = board.legal_moves
        if not legal_moves:
            break
        move = np.random.choice(legal_moves)
        board.make_move(move)
    
    # 根据结果设置节点价值
    state = board.get_state()
    if state.is_checkmate:
        # 将死：最后走棋的一方获胜
        value = 1.0 if not board.turn else -1.0
    elif state.is_stalemate or state.is_insufficient_material or state.is_fifty_moves:
        # 和局
        value = 0.0
    else:
        value = 0.0
    
    node.Q = value
    node.N = 1


def get_root_policy(root: MCTSNode) -> np.ndarray:
    """
    从根节点提取策略分布 π(s)
    
    π(a|s) = N(s,a)^τ / Σ_b N(s,b)^τ
    
    Args:
        root: 根节点
        
    Returns:
        [4608] 策略概率分布
    """
    pi = np.zeros(4608, dtype=np.float32)
    
    if root.N == 0:
        # 均匀分布
        n_moves = len(root.legal_moves)
        if n_moves > 0:
            for move in root.legal_moves:
                idx = _move_to_index(move)
                if idx < 4608:
                    pi[idx] = 1.0 / n_moves
        return pi
    
    # 温度参数 τ=1（不使用温度缩放）
    tau = 1.0
    
    total_count = 0
    for move, child in root.children.items():
        idx = _move_to_index(move)
        if idx < 4608:
            count = child.N ** tau
            pi[idx] = count
            total_count += count
    
    # 归一化
    if total_count > 0:
        pi /= total_count
    
    return pi


def _get_model_prior(model, board: Board) -> Tuple[np.ndarray, float]:
    """
    从神经网络获取先验概率和价值
    
    Args:
        model: 神经网络模型
        board: 棋盘状态
        
    Returns:
        priors: [4608] 策略先验
        value: 标量价值估计
    """
    from model.encoder import encode_board
    import torch
    
    # 编码棋盘状态
    state_tensor = encode_board(board)
    
    # 添加 batch 维度
    if len(state_tensor.shape) == 3:
        state_tensor = state_tensor.unsqueeze(0)
    
    # 模型推理
    with torch.no_grad():
        policy_logits, value = model(state_tensor)
    
    # 处理策略输出
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    value_scalar = value.squeeze().item()
    
    return policy_probs, value_scalar


def _move_to_index(move: chess.Move) -> int:
    """
    将 chess.Move 转换为策略向量索引
    
    策略向量大小：4608 = 64 格 × 72 种走法
    72 种走法包括：
    -  queen-like moves (56): 8 directions × 7 distances
    -  knight moves (8)
    -  underpromotions (6): 3 piece types × 2 directions (capture/non-capture)
    
    Args:
        move: chess.Move 对象
        
    Returns:
        策略向量中的索引 (0-4607)
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion
    
    # 计算方向向量
    dx = chess.square_file(to_square) - chess.square_file(from_square)
    dy = chess.square_rank(to_square) - chess.square_rank(from_square)
    
    base_index = from_square * 72
    
    # 检查是否是升变
    if promotion is not None:
        # 升变走法
        knight_dx = abs(dx)
        knight_dy = dy
        
        if knight_dx == 1 and knight_dy == 2:
            # 马步升变（理论上不应该发生）
            offset = 56 + _knight_offset(dx, dy)
        elif knight_dx == 2 and knight_dy == 1:
            offset = 56 + _knight_offset(dx, dy)
        else:
            # 后翼/王翼升变
            direction = 0 if dx < 0 else (1 if dx > 0 else 2)
            piece_type = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promotion]
            offset = 56 + 8 + direction * 2 + (0 if promotion == chess.QUEEN else piece_type)
        
        return base_index + offset
    
    # 普通走法
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        # 直线或斜线走法（后、车、象）
        direction = _get_direction(dx, dy)
        distance = max(abs(dx), abs(dy)) - 1
        offset = direction * 7 + distance
        return base_index + offset
    else:
        # 马步
        offset = 56 + _knight_offset(dx, dy)
        return base_index + offset


def _get_direction(dx: int, dy: int) -> int:
    """获取方向索引 (0-7)"""
    directions = [
        (0, 1),   # N
        (1, 1),   # NE
        (1, 0),   # E
        (1, -1),  # SE
        (0, -1),  # S
        (-1, -1), # SW
        (-1, 0),  # W
        (-1, 1),  # NW
    ]
    for i, (ddx, ddy) in enumerate(directions):
        if dx == ddx and dy == ddy:
            return i
    return 0


def _knight_offset(dx: int, dy: int) -> int:
    """获取马步偏移量 (0-7)"""
    knight_moves = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]
    for i, (ddx, ddy) in enumerate(knight_moves):
        if dx == ddx and dy == ddy:
            return i
    return 0
