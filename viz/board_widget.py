"""
单盘棋盘渲染组件
负责单个棋盘的绘制、棋子摆放、动画效果
"""

import pygame
import chess
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .assets import AssetLoader


@dataclass
class SquareHighlight:
    """格子高亮信息"""
    square: int  # chess square (0-63)
    color: Tuple[int, int, int, int]  # RGBA


@dataclass
class MoveAnimation:
    """走子动画信息"""
    piece: str  # piece key (e.g., 'white-pawn')
    start_pos: Tuple[int, int]  # 起始像素坐标
    end_pos: Tuple[int, int]    # 结束像素坐标
    progress: float             # 动画进度 (0.0 - 1.0)
    size: Tuple[int, int]       # 棋子尺寸


class BoardWidget:
    """
    棋盘组件
    渲染单个 8x8 国际象棋棋盘
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        size: int,
        asset_loader: AssetLoader,
        orientation: bool = chess.WHITE,  # True=白方在下，False=黑方在下
    ):
        """
        初始化棋盘组件
        
        Args:
            x: 组件左上角 X 坐标
            y: 组件左上角 Y 坐标
            size: 棋盘边长（像素）
            asset_loader: 资源加载器
            orientation: 棋盘方向
        """
        self.x = x
        self.y = y
        self.size = size
        self.asset_loader = asset_loader
        self.orientation = orientation
        
        self.square_size = size // 8
        
        # 高亮和动画
        self._highlights: List[SquareHighlight] = []
        self._animations: List[MoveAnimation] = []
        
        # 选中状态
        self._selected_square: Optional[int] = None
        self._legal_moves: List[chess.Move] = []
        
        # 缓存
        self._board_surface: Optional[pygame.Surface] = None
    
    def _square_to_coords(self, square: int) -> Tuple[int, int]:
        """
        将 chess square 转换为网格坐标 (row, col)
        
        Returns:
            (row, col): 行和列 (0-7)
        """
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        
        if not self.orientation:
            # 黑方视角翻转
            row = 7 - row
            col = 7 - col
        
        return row, col
    
    def _coords_to_square(self, row: int, col: int) -> int:
        """
        将网格坐标转换为 chess square
        
        Returns:
            chess square (0-63)
        """
        if not self.orientation:
            row = 7 - row
            col = 7 - col
        
        rank = 7 - row
        file = col
        
        return chess.square(file, rank)
    
    def _square_to_pixel(self, square: int) -> Tuple[int, int]:
        """
        将 chess square 转换为像素坐标（左上角）
        
        Returns:
            (x, y): 像素坐标
        """
        row, col = self._square_to_coords(square)
        
        pixel_x = self.x + col * self.square_size
        pixel_y = self.y + row * self.square_size
        
        return pixel_x, pixel_y
    
    def _pixel_to_square(self, pixel_x: int, pixel_y: int) -> Optional[int]:
        """
        将像素坐标转换为 chess square
        
        Returns:
            chess square 或 None（如果在棋盘外）
        """
        rel_x = pixel_x - self.x
        rel_y = pixel_y - self.y
        
        if not (0 <= rel_x < self.size and 0 <= rel_y < self.size):
            return None
        
        col = rel_x // self.square_size
        row = rel_y // self.square_size
        
        return self._coords_to_square(row, col)
    
    def draw(self, surface: pygame.Surface, board: chess.Board) -> None:
        """
        绘制棋盘和棋子
        
        Args:
            surface: Pygame 表面
            board: python-chess Board 对象
        """
        # 绘制棋盘背景
        self._draw_board_background(surface)
        
        # 绘制高亮
        self._draw_highlights(surface)
        
        # 绘制选中格子和合法走法提示
        self._draw_selection(surface, board)
        
        # 绘制棋子
        self._draw_pieces(surface, board)
        
        # 绘制动画
        self._draw_animations(surface)
    
    def _draw_board_background(self, surface: pygame.Surface) -> None:
        """绘制棋盘背景"""
        # 尝试使用加载的图片
        board_image = self.asset_loader.get_board("rect-8x8")
        
        if board_image:
            # 缩放图片到指定大小
            scaled_image = pygame.transform.smoothscale(board_image, (self.size, self.size))
            surface.blit(scaled_image, (self.x, self.y))
        else:
            # 程序化生成棋盘
            colored_board = self.asset_loader.create_colored_board(self.size)
            surface.blit(colored_board, (self.x, self.y))
    
    def _draw_highlights(self, surface: pygame.Surface) -> None:
        """绘制高亮区域"""
        for highlight in self._highlights:
            pixel_x, pixel_y = self._square_to_pixel(highlight.square)
            
            # 创建半透明覆盖层
            overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            overlay.fill(highlight.color)
            surface.blit(overlay, (pixel_x, pixel_y))
    
    def _draw_selection(
        self,
        surface: pygame.Surface,
        board: chess.Board
    ) -> None:
        """绘制选中格子和合法走法提示"""
        if self._selected_square is None:
            return
        
        # 高亮选中格子
        pixel_x, pixel_y = self._square_to_pixel(self._selected_square)
        
        selection_color = (100, 255, 100, 150)  # 浅绿色半透明
        overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        overlay.fill(selection_color)
        surface.blit(overlay, (pixel_x, pixel_y))
        
        # 高亮合法走法目标格
        for move in self._legal_moves:
            if move.from_square == self._selected_square:
                target_x, target_y = self._square_to_pixel(move.to_square)
                
                # 画圆圈提示
                center_x = target_x + self.square_size // 2
                center_y = target_y + self.square_size // 2
                
                if board.piece_at(move.to_square):
                    # 吃子：红色圆环
                    pygame.draw.circle(
                        surface,
                        (255, 50, 50),
                        (center_x, center_y),
                        self.square_size // 2 - 4,
                        width=4
                    )
                else:
                    # 移动：绿色小圆点
                    pygame.draw.circle(
                        surface,
                        (100, 255, 100),
                        (center_x, center_y),
                        self.square_size // 6
                    )
    
    def _draw_pieces(self, surface: pygame.Surface, board: chess.Board) -> None:
        """绘制所有棋子"""
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # 获取棋子图片键名
            color = 'white' if piece.color == chess.WHITE else 'black'
            piece_type = chess.piece_name(piece.piece_type)
            piece_key = f"{color}-{piece_type}"
            
            # 获取并缩放棋子图片
            piece_image = self.asset_loader.resize_piece(
                piece_key,
                (self.square_size, self.square_size)
            )
            
            if piece_image:
                pixel_x, pixel_y = self._square_to_pixel(square)
                surface.blit(piece_image, (pixel_x, pixel_y))
    
    def _draw_animations(self, surface: pygame.Surface) -> None:
        """绘制走子动画"""
        for anim in self._animations:
            # 插值计算当前位置
            current_x = int(anim.start_pos[0] + (anim.end_pos[0] - anim.start_pos[0]) * anim.progress)
            current_y = int(anim.start_pos[1] + (anim.end_pos[1] - anim.start_pos[1]) * anim.progress)
            
            # 获取棋子图片
            piece_image = self.asset_loader.resize_piece(anim.piece, anim.size)
            
            if piece_image:
                surface.blit(piece_image, (current_x, current_y))
    
    def handle_click(self, pixel_x: int, pixel_y: int, board: chess.Board) -> Optional[int]:
        """
        处理鼠标点击
        
        Args:
            pixel_x: 鼠标 X 坐标
            pixel_y: 鼠标 Y 坐标
            board: 当前棋盘状态
            
        Returns:
            选中的 square 或 None
        """
        square = self._pixel_to_square(pixel_x, pixel_y)
        
        if square is None:
            return None
        
        # 如果已经选中了一个格子
        if self._selected_square is not None:
            # 检查是否是合法走法
            for move in self._legal_moves:
                if move.from_square == self._selected_square and move.to_square == square:
                    # 这是一个合法走法，清除选中并返回原选中格
                    selected = self._selected_square
                    self.clear_selection()
                    return selected
            
            # 点击了其他格子，更新选中
            if board.piece_at(square) is not None:
                self._selected_square = square
                self._update_legal_moves(board)
            else:
                self.clear_selection()
        else:
            # 首次点击，如果有棋子则选中
            if board.piece_at(square) is not None:
                self._selected_square = square
                self._update_legal_moves(board)
        
        return None
    
    def _update_legal_moves(self, board: chess.Board) -> None:
        """更新合法走法列表"""
        if self._selected_square is None:
            self._legal_moves = []
            return
        
        self._legal_moves = [
            move for move in board.legal_moves
            if move.from_square == self._selected_square
        ]
    
    def clear_selection(self) -> None:
        """清除选中状态"""
        self._selected_square = None
        self._legal_moves = []
    
    def add_highlight(self, square: int, color: Tuple[int, int, int, int]) -> None:
        """添加高亮"""
        self._highlights.append(SquareHighlight(square=square, color=color))
    
    def clear_highlights(self) -> None:
        """清除所有高亮"""
        self._highlights.clear()
    
    def start_move_animation(
        self,
        from_square: int,
        to_square: int,
        piece: str,
        duration: float = 0.3
    ) -> None:
        """
        开始走子动画
        
        Args:
            from_square: 起始格子
            to_square: 目标格子
            piece: 棋子键名
            duration: 动画时长（秒）
        """
        start_pos = self._square_to_pixel(from_square)
        end_pos = self._square_to_pixel(to_square)
        
        self._animations.append(MoveAnimation(
            piece=piece,
            start_pos=start_pos,
            end_pos=end_pos,
            progress=0.0,
            size=(self.square_size, self.square_size)
        ))
    
    def update_animations(self, dt: float) -> List[MoveAnimation]:
        """
        更新动画进度
        
        Args:
            dt: 时间增量（秒）
            
        Returns:
            已完成的动画列表
        """
        completed = []
        remaining = []
        
        for anim in self._animations:
            anim.progress += dt / 0.3  # 默认 0.3 秒完成
            
            if anim.progress >= 1.0:
                completed.append(anim)
            else:
                remaining.append(anim)
        
        self._animations = remaining
        return completed
    
    def set_orientation(self, orientation: bool) -> None:
        """设置棋盘方向"""
        self.orientation = orientation
    
    def get_square_under_mouse(self, pixel_x: int, pixel_y: int) -> Optional[int]:
        """获取鼠标下的格子"""
        return self._pixel_to_square(pixel_x, pixel_y)
