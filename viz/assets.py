"""
资源加载器 - 加载棋盘和棋子图片
"""

import os
from pathlib import Path
from typing import Dict, Optional
import pygame


class AssetLoader:
    """
    游戏资源加载器
    负责加载和管理棋盘、棋子等图片资源
    """
    
    def __init__(
        self,
        pieces_dir: str = "resources/png/pieces",
        board_dir: str = "resources/png/board",
    ):
        """
        初始化资源加载器
        
        Args:
            pieces_dir: 棋子图片目录
            board_dir: 棋盘图片目录
        """
        self.pieces_dir = Path(pieces_dir)
        self.board_dir = Path(board_dir)
        
        self._board_images: Dict[str, pygame.Surface] = {}
        self._piece_images: Dict[str, pygame.Surface] = {}
        
        # 棋子文件名映射
        self._piece_filenames = {
            'white-pawn': 'white-pawn.png',
            'white-rook': 'white-rook.png',
            'white-knight': 'white-knight.png',
            'white-bishop': 'white-bishop.png',
            'white-queen': 'white-queen.png',
            'white-king': 'white-king.png',
            'black-pawn': 'black-pawn.png',
            'black-rook': 'black-rook.png',
            'black-knight': 'black-knight.png',
            'black-bishop': 'black-bishop.png',
            'black-queen': 'black-queen.png',
            'black-king': 'black-king.png',
        }
    
    def load_all(self) -> None:
        """加载所有资源"""
        self.load_board()
        self.load_pieces()
    
    def load_board(self, name: str = "rect-8x8") -> Optional[pygame.Surface]:
        """
        加载棋盘图片
        
        Args:
            name: 棋盘图片文件名（不含扩展名）
            
        Returns:
            加载的 Surface，失败返回 None
        """
        if name in self._board_images:
            return self._board_images[name]
        
        image_path = self.board_dir / f"{name}.png"
        
        if not image_path.exists():
            print(f"Warning: Board image not found: {image_path}")
            return None
        
        try:
            image = pygame.image.load(str(image_path)).convert()
            self._board_images[name] = image
            return image
        except pygame.error as e:
            print(f"Error loading board image: {e}")
            return None
    
    def load_piece(self, piece_key: str) -> Optional[pygame.Surface]:
        """
        加载单个棋子图片
        
        Args:
            piece_key: 棋子键名 (如 'white-pawn')
            
        Returns:
            加载的 Surface，失败返回 None
        """
        if piece_key in self._piece_images:
            return self._piece_images[piece_key]
        
        if piece_key not in self._piece_filenames:
            print(f"Warning: Unknown piece key: {piece_key}")
            return None
        
        filename = self._piece_filenames[piece_key]
        image_path = self.pieces_dir / filename
        
        if not image_path.exists():
            print(f"Warning: Piece image not found: {image_path}")
            return None
        
        try:
            image = pygame.image.load(str(image_path)).convert_alpha()
            self._piece_images[piece_key] = image
            return image
        except pygame.error as e:
            print(f"Error loading piece image: {e}")
            return None
    
    def load_pieces(self) -> None:
        """加载所有棋子图片"""
        for piece_key in self._piece_filenames.keys():
            self.load_piece(piece_key)
    
    def get_board(self, name: str = "rect-8x8") -> Optional[pygame.Surface]:
        """获取已加载的棋盘图片"""
        return self._board_images.get(name)
    
    def get_piece(self, piece_key: str) -> Optional[pygame.Surface]:
        """获取已加载的棋子图片"""
        return self._piece_images.get(piece_key)
    
    def resize_piece(self, piece_key: str, size: tuple) -> Optional[pygame.Surface]:
        """
        获取调整大小后的棋子图片
        
        Args:
            piece_key: 棋子键名
            size: (width, height) 目标尺寸
            
        Returns:
            调整后的 Surface
        """
        original = self.get_piece(piece_key)
        if not original:
            return None
        
        return pygame.transform.smoothscale(original, size)
    
    def create_colored_board(
        self,
        size: int,
        light_color: tuple = (240, 217, 181),
        dark_color: tuple = (181, 136, 99),
    ) -> pygame.Surface:
        """
        程序化生成棋盘背景（如果没有图片资源）
        
        Args:
            size: 棋盘边长（像素）
            light_color: 浅色格颜色 (R, G, B)
            dark_color: 深色格颜色 (R, G, B)
            
        Returns:
            生成的棋盘 Surface
        """
        surface = pygame.Surface((size, size))
        square_size = size // 8
        
        for row in range(8):
            for col in range(8):
                # 国际象棋棋盘颜色交替
                is_light = (row + col) % 2 == 0
                color = light_color if is_light else dark_color
                
                x = col * square_size
                y = row * square_size
                
                pygame.draw.rect(surface, color, (x, y, square_size, square_size))
        
        return surface
