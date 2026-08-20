"""
Pygame 主循环 - 单窗口多盘网格布局
"""

import pygame
import asyncio
from typing import Dict, List, Optional, Tuple, Callable
import chess

from engine.game import Game, GameState, GameResult
from engine.scheduler import MatchScheduler
from engine.scoring import ScoringSystem
from .assets import AssetLoader
from .board_widget import BoardWidget
from .scoreboard import ScoreboardPanel


class VisualizationApp:
    """
    可视化应用主类
    Pygame 主循环，管理多棋盘网格布局和 UI 交互
    """
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        grid_rows: int = 2,
        grid_cols: int = 2,
        scheduler: Optional[MatchScheduler] = None,
    ):
        """
        初始化可视化应用
        
        Args:
            width: 窗口宽度
            height: 窗口高度
            grid_rows: 棋盘网格行数
            grid_cols: 棋盘网格列数
            scheduler: 比赛调度器
        """
        self.width = width
        self.height = height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Pygame 初始化
        pygame.init()
        pygame.display.set_caption("ChessRL v1.0 - P0 Demo")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # 资源加载器
        self.asset_loader = AssetLoader()
        self.asset_loader.load_all()
        
        # 调度器和计分系统
        self.scheduler = scheduler
        self.scoring_system = scheduler.scoring_system if scheduler else ScoringSystem()
        
        # 棋盘组件列表
        self._board_widgets: Dict[str, BoardWidget] = {}
        
        # 计分板面板
        scoreboard_width = 300
        self._scoreboard = ScoreboardPanel(
            x=width - scoreboard_width - 10,
            y=10,
            width=scoreboard_width,
            height=height - 20,
            scoring_system=self.scoring_system,
        )
        
        # 游戏到棋盘的映射
        self._game_board_map: Dict[str, str] = {}  # game_id -> board_widget_key
        
        # 计算棋盘区域大小
        available_width = width - scoreboard_width - 40
        available_height = height - 60  # 留出顶部栏空间
        
        self.board_size = min(
            available_width // grid_cols,
            available_height // grid_rows,
            400  # 最大棋盘尺寸
        )
        
        # 选中状态
        self._selected_game_id: Optional[str] = None
        self._selected_square: Optional[int] = None
        
        # 运行状态
        self._running = False
        self._paused = False
        
        # 回调函数
        self._on_move_made: Optional[Callable] = None
        self._on_human_move_request: Optional[Callable[[str, int], None]] = None
    
    def setup_boards(self) -> None:
        """设置棋盘网格布局"""
        self._board_widgets.clear()
        
        start_x = 20
        start_y = 50
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                key = f"board_{row}_{col}"
                x = start_x + col * (self.board_size + 10)
                y = start_y + row * (self.board_size + 10)
                
                widget = BoardWidget(
                    x=x,
                    y=y,
                    size=self.board_size,
                    asset_loader=self.asset_loader,
                )
                
                self._board_widgets[key] = widget
    
    def register_game(self, game: Game) -> Optional[str]:
        """
        注册一个游戏到显示网格
        
        Returns:
            board_widget_key 或 None（无空闲位置）
        """
        # 查找空闲的棋盘位置
        for key, widget in self._board_widgets.items():
            if key not in self._game_board_map.values():
                self._game_board_map[game.game_id] = key
                return key
        
        return None
    
    def unregister_game(self, game_id: str) -> None:
        """注销游戏"""
        if game_id in self._game_board_map:
            del self._game_board_map[game_id]
    
    def _draw_header(self) -> None:
        """绘制顶部栏"""
        header_rect = pygame.Rect(0, 0, self.width, 40)
        pygame.draw.rect(self.screen, (30, 30, 50), header_rect)
        
        font = pygame.font.Font(None, 32)
        title = font.render("ChessRL v1.0 - P0 Demo", True, (255, 255, 255))
        self.screen.blit(title, (15, 10))
        
        # 状态指示
        status_font = pygame.font.Font(None, 24)
        if self._paused:
            status = status_font.render("PAUSED", True, (255, 200, 100))
            self.screen.blit(status, (self.width - 100, 12))
    
    def _update_teacher_student_roles(self) -> None:
        """更新 Teacher/Student 角色显示"""
        rankings = self.scoring_system.get_rankings()
        
        if len(rankings) >= 2:
            n_agents = len(rankings)
            half = n_agents // 2
            
            teacher_ids = [agent_id for agent_id, _ in rankings[:half]]
            student_ids = [agent_id for agent_id, _ in rankings[half:]]
            
            self._scoreboard.set_roles(teacher_ids, student_ids)
    
    def handle_events(self) -> bool:
        """
        处理事件
        
        Returns:
            bool: 是否继续运行
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_p:
                    self._paused = not self._paused
                elif event.key == pygame.K_r:
                    # 重置（待实现）
                    pass
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not self._paused:
                if event.button == 1:  # 左键
                    self._handle_mouse_click(event.pos)
        
        return True
    
    def _handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        """处理鼠标点击"""
        mouse_x, mouse_y = pos
        
        # 检查是否点击了计分板
        clicked_agent = self._scoreboard.get_clicked_agent(mouse_x, mouse_y)
        if clicked_agent:
            print(f"Clicked agent: {clicked_agent}")
            return
        
        # 检查是否点击了某个棋盘
        for game_id, board_key in self._game_board_map.items():
            widget = self._board_widgets.get(board_key)
            if not widget:
                continue
            
            square = widget.get_square_under_mouse(mouse_x, mouse_y)
            if square is not None:
                # 找到对应的游戏
                if self.scheduler:
                    for game in self.scheduler.get_active_games():
                        if game.game_id == game_id:
                            # 处理点击
                            from_square = widget.handle_click(mouse_x, mouse_y, game.board.internal_board)
                            if from_square is not None:
                                # 玩家选择了走法
                                if self._on_human_move_request:
                                    self._on_human_move_request(game_id, from_square)
                            break
    
    def draw(self) -> None:
        """渲染所有元素"""
        self.screen.fill((20, 20, 30))
        
        # 绘制顶部栏
        self._draw_header()
        
        # 绘制所有棋盘
        for game_id, board_key in self._game_board_map.items():
            widget = self._board_widgets.get(board_key)
            if not widget:
                continue
            
            # 获取游戏棋盘状态
            if self.scheduler:
                for game in self.scheduler.get_active_games():
                    if game.game_id == game_id:
                        widget.draw(self.screen, game.board.internal_board)
                        break
        
        # 绘制计分板
        self._update_teacher_student_roles()
        self._scoreboard.draw(self.screen)
        
        pygame.display.flip()
    
    def update(self, dt: float) -> None:
        """更新逻辑（动画等）"""
        # 更新棋盘动画
        for widget in self._board_widgets.values():
            widget.update_animations(dt)
    
    async def run_async(self) -> None:
        """异步主循环"""
        self._running = True
        self.setup_boards()
        
        while self._running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS
            
            if not self.handle_events():
                self._running = False
                break
            
            if not self._paused:
                self.update(dt)
            
            self.draw()
            
            # 让出控制权给其他协程
            await asyncio.sleep(0)
        
        self.quit()
    
    def run(self) -> None:
        """同步主循环（用于测试）"""
        self._running = True
        self.setup_boards()
        
        while self._running:
            dt = self.clock.tick(60) / 1000.0
            
            if not self.handle_events():
                self._running = False
                break
            
            if not self._paused:
                self.update(dt)
            
            self.draw()
    
    def quit(self) -> None:
        """退出应用"""
        pygame.quit()
    
    def set_callbacks(
        self,
        on_move_made: Optional[Callable] = None,
        on_human_move_request: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """设置回调函数"""
        self._on_move_made = on_move_made
        self._on_human_move_request = on_human_move_request


# 便捷函数：创建并运行演示
async def run_demo(
    scheduler: MatchScheduler,
    width: int = 1200,
    height: int = 800,
) -> None:
    """
    运行演示模式
    
    Args:
        scheduler: 比赛调度器
        width: 窗口宽度
        height: 窗口高度
    """
    app = VisualizationApp(
        width=width,
        height=height,
        scheduler=scheduler,
    )
    
    await app.run_async()
