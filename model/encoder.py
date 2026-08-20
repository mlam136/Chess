"""
棋盘状态编码器 - 将棋盘状态转换为模型输入张量

Input: chess.Board
Output: [16, 8, 8] 张量

通道定义：
- 0-5:   白方 (K, Q, R, B, N, P) one-hot 平面
- 6-11:  黑方 (K, Q, R, B, N, P) one-hot 平面
- 12:    当前回合 (白=1, 黑=0) 全 1/全 0
- 13:    合法走法目标格 mask
- 14:    合法走法源格 mask
- 15:    特殊标记（可吃过路兵/可王车易位）
"""

import torch
import numpy as np
import chess
from typing import Optional

from engine.board import Board


# 棋子到通道映射
PIECE_TO_CHANNEL = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
    chess.PAWN: 5,
}


def encode_board(board: Board) -> torch.Tensor:
    """
    将棋盘状态编码为模型输入张量
    
    Args:
        board: Board 对象
        
    Returns:
        [16, 8, 8] 浮点张量
    """
    internal_board = board.internal_board
    
    # 初始化 16 通道张量
    tensor = np.zeros((16, 8, 8), dtype=np.float32)
    
    # 填充棋子位置
    _encode_pieces(internal_board, tensor)
    
    # 当前回合
    tensor[12, :, :] = 1.0 if internal_board.turn else 0.0
    
    # 合法走法 mask
    _encode_legal_moves(internal_board, tensor)
    
    # 特殊标记
    _encode_special_features(internal_board, tensor)
    
    return torch.from_numpy(tensor)


def _encode_pieces(board: chess.Board, tensor: np.ndarray) -> None:
    """
    编码棋子位置到张量
    
    Args:
        board: chess.Board 对象
        tensor: [16, 8, 8] 张量
    """
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # 确定通道
        channel = PIECE_TO_CHANNEL[piece.piece_type]
        if piece.color == chess.BLACK:
            channel += 6  # 黑方棋子偏移 6 个通道
        
        tensor[channel, rank, file] = 1.0


def _encode_legal_moves(board: chess.Board, tensor: np.ndarray) -> None:
    """
    编码合法走法 mask
    
    Args:
        board: chess.Board 对象
        tensor: [16, 8, 8] 张量
    """
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)
        to_file = chess.square_file(to_square)
        to_rank = chess.square_rank(to_square)
        
        # 源格 mask
        tensor[14, from_rank, from_file] = 1.0
        # 目标格 mask
        tensor[13, to_rank, to_file] = 1.0


def _encode_special_features(board: chess.Board, tensor: np.ndarray) -> None:
    """
    编码特殊特征（过路兵、王车易位）
    
    Args:
        board: chess.Board 对象
        tensor: [16, 8, 8] 张量
    """
    # 过路兵
    if board.has_legal_en_passant():
        ep_square = board.ep_square
        if ep_square is not None:
            ep_file = chess.square_file(ep_square)
            ep_rank = chess.square_rank(ep_square)
            tensor[15, ep_rank, ep_file] = 1.0
    
    # 王车易位
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[15, 0, 7] = 1.0  # 白方王翼
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[15, 0, 0] = 1.0  # 白方后翼
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, 7, 7] = 1.0  # 黑方王翼
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, 7, 0] = 1.0  # 黑方后翼


def encode_board_batch(boards: list) -> torch.Tensor:
    """
    批量编码多个棋盘状态
    
    Args:
        boards: Board 对象列表
        
    Returns:
        [B, 16, 8, 8] 批处理张量
    """
    tensors = [encode_board(board) for board in boards]
    return torch.stack(tensors)


def decode_board_tensor(tensor: torch.Tensor) -> dict:
    """
    解码张量为可读信息（用于调试）
    
    Args:
        tensor: [16, 8, 8] 或 [B, 16, 8, 8] 张量
        
    Returns:
        包含各通道信息的字典
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取 batch 第一个
    
    tensor_np = tensor.cpu().numpy()
    
    result = {
        'white_pieces': {},
        'black_pieces': {},
        'turn': 'white' if tensor_np[12].sum() > 0 else 'black',
        'legal_moves_from': [],
        'legal_moves_to': [],
    }
    
    # 解码白方棋子
    for piece_type, channel in PIECE_TO_CHANNEL.items():
        positions = np.where(tensor_np[channel] > 0.5)
        for rank, file in zip(positions[0], positions[1]):
            pos_name = chess.square_name(chess.square(file, rank))
            result['white_pieces'][pos_name] = chess.piece_symbol(piece_type).upper()
    
    # 解码黑方棋子
    for piece_type, channel in PIECE_TO_CHANNEL.items():
        positions = np.where(tensor_np[channel + 6] > 0.5)
        for rank, file in zip(positions[0], positions[1]):
            pos_name = chess.square_name(chess.square(file, rank))
            result['black_pieces'][pos_name] = chess.piece_symbol(piece_type).lower()
    
    # 解码合法走法
    from_positions = np.where(tensor_np[14] > 0.5)
    to_positions = np.where(tensor_np[13] > 0.5)
    
    for rank, file in zip(from_positions[0], from_positions[1]):
        result['legal_moves_from'].append(chess.square_name(chess.square(file, rank)))
    
    for rank, file in zip(to_positions[0], to_positions[1]):
        result['legal_moves_to'].append(chess.square_name(chess.square(file, rank)))
    
    return result


if __name__ == '__main__':
    # 测试编码器
    board = Board()
    tensor = encode_board(board)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    # 解码查看
    info = decode_board_tensor(tensor)
    print(f"\nTurn: {info['turn']}")
    print(f"White pieces: {info['white_pieces']}")
    print(f"Black pieces: {info['black_pieces']}")
    print(f"Legal moves from: {info['legal_moves_from'][:5]}...")
    print(f"Legal moves to: {info['legal_moves_to'][:5]}...")
