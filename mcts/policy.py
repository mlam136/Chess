"""
策略提取工具函数
"""

import numpy as np
from typing import List
import chess


def extract_policy(mcts_visits: dict, legal_moves: List[chess.Move]) -> np.ndarray:
    """
    从 MCTS 访问次数中提取策略分布
    
    Args:
        mcts_visits: {move: visit_count} 字典
        legal_moves: 合法走法列表
        
    Returns:
        [4608] 策略概率分布
    """
    pi = np.zeros(4608, dtype=np.float32)
    
    total_visits = sum(mcts_visits.values())
    
    if total_visits == 0:
        # 均匀分布
        n_moves = len(legal_moves)
        if n_moves > 0:
            for move in legal_moves:
                idx = _move_to_index(move)
                if idx < 4608:
                    pi[idx] = 1.0 / n_moves
        return pi
    
    # 按访问次数分配概率
    for move, count in mcts_visits.items():
        idx = _move_to_index(move)
        if idx < 4608:
            pi[idx] = count / total_visits
    
    return pi


def policy_to_move(pi: np.ndarray, legal_moves: List[chess.Move], 
                   temperature: float = 1.0) -> chess.Move:
    """
    从策略分布中选择走法
    
    Args:
        pi: [4608] 策略概率分布
        legal_moves: 合法走法列表
        temperature: 温度参数（>1 增加随机性，<1 增加确定性）
        
    Returns:
        选中的走法
    """
    # 获取合法走法的概率
    move_probs = []
    valid_moves = []
    
    for move in legal_moves:
        idx = _move_to_index(move)
        if idx < 4608:
            prob = pi[idx]
            move_probs.append(prob)
            valid_moves.append(move)
    
    if not valid_moves:
        return None
    
    # 应用温度缩放
    probs = np.array(move_probs, dtype=np.float64)
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    
    # 归一化
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum
    else:
        # 均匀分布
        probs = np.ones(len(probs)) / len(probs)
    
    # 采样
    selected_idx = np.random.choice(len(valid_moves), p=probs)
    return valid_moves[selected_idx]


def best_move_from_policy(pi: np.ndarray, legal_moves: List[chess.Move]) -> chess.Move:
    """
    从策略分布中选择概率最高的走法（贪婪）
    
    Args:
        pi: [4608] 策略概率分布
        legal_moves: 合法走法列表
        
    Returns:
        概率最高的走法
    """
    best_prob = -1
    best_move = None
    
    for move in legal_moves:
        idx = _move_to_index(move)
        if idx < 4608:
            prob = pi[idx]
            if prob > best_prob:
                best_prob = prob
                best_move = move
    
    return best_move


def _move_to_index(move: chess.Move) -> int:
    """
    将 chess.Move 转换为策略向量索引
    
    Args:
        move: chess.Move 对象
        
    Returns:
        策略向量中的索引 (0-4607)
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion
    
    dx = chess.square_file(to_square) - chess.square_file(from_square)
    dy = chess.square_rank(to_square) - chess.square_rank(from_square)
    
    base_index = from_square * 72
    
    if promotion is not None:
        knight_dx = abs(dx)
        knight_dy = dy
        
        if knight_dx == 1 and knight_dy == 2:
            offset = 56 + _knight_offset(dx, dy)
        elif knight_dx == 2 and knight_dy == 1:
            offset = 56 + _knight_offset(dx, dy)
        else:
            direction = 0 if dx < 0 else (1 if dx > 0 else 2)
            piece_type = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}.get(promotion, 0)
            offset = 56 + 8 + direction * 2 + (0 if promotion == chess.QUEEN else piece_type)
        
        return base_index + offset
    
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        direction = _get_direction(dx, dy)
        distance = max(abs(dx), abs(dy)) - 1
        offset = direction * 7 + distance
        return base_index + offset
    else:
        offset = 56 + _knight_offset(dx, dy)
        return base_index + offset


def _get_direction(dx: int, dy: int) -> int:
    """获取方向索引 (0-7)"""
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
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
