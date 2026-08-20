"""
分数/身份面板渲染
显示所有智能体的得分、排名、角色信息
"""

import pygame
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from engine.scoring import ScoringSystem, AgentScore


@dataclass
class AgentDisplayInfo:
    """智能体显示信息"""
    agent_id: str
    score: float
    games_played: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    role: str  # "Teacher" / "Student" / "-"


class ScoreboardPanel:
    """
    计分板面板
    显示所有智能体的统计信息和排名
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        scoring_system: ScoringSystem,
    ):
        """
        初始化计分板
        
        Args:
            x: 左上角 X 坐标
            y: 左上角 Y 坐标
            width: 面板宽度
            height: 面板高度
            scoring_system: 计分系统
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scoring_system = scoring_system
        
        # 样式配置
        self.bg_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.header_color = (60, 60, 80)
        self.row_colors = [(50, 50, 50), (45, 45, 45)]  # 交替行颜色
        
        # 字体（在 draw 时初始化）
        self._font_header: Optional[pygame.font.Font] = None
        self._font_normal: Optional[pygame.font.Font] = None
        
        # 列宽配置
        self.columns = [
            ("Rank", 50),
            ("Agent", 100),
            ("Score", 70),
            ("Games", 60),
            ("W/D/L", 90),
            ("Win%", 60),
            ("Role", 80),
        ]
        
        # Teacher/Student 分配
        self._teacher_agents: set = set()
        self._student_agents: set = set()
    
    def _ensure_fonts(self) -> None:
        """确保字体已初始化"""
        if self._font_header is None:
            self._font_header = pygame.font.Font(None, 28)
            self._font_normal = pygame.font.Font(None, 24)
    
    def set_roles(self, teacher_ids: List[str], student_ids: List[str]) -> None:
        """设置 Teacher/Student 角色分配"""
        self._teacher_agents = set(teacher_ids)
        self._student_agents = set(student_ids)
    
    def _get_agent_info(self, agent_id: str, rank: int) -> AgentDisplayInfo:
        """获取智能体显示信息"""
        score_data = self.scoring_system.get_score(agent_id)
        
        if score_data:
            return AgentDisplayInfo(
                agent_id=agent_id,
                score=self.scoring_system.get_window_score(agent_id),
                games_played=score_data.games_played,
                wins=score_data.wins,
                draws=score_data.draws,
                losses=score_data.losses,
                win_rate=score_data.win_rate * 100,
                role=self._get_role(agent_id),
            )
        else:
            return AgentDisplayInfo(
                agent_id=agent_id,
                score=0.0,
                games_played=0,
                wins=0,
                draws=0,
                losses=0,
                win_rate=0.0,
                role=self._get_role(agent_id),
            )
    
    def _get_role(self, agent_id: str) -> str:
        """获取智能体角色"""
        if agent_id in self._teacher_agents:
            return "Teacher"
        elif agent_id in self._student_agents:
            return "Student"
        return "-"
    
    def draw(self, surface: pygame.Surface) -> None:
        """
        绘制计分板
        
        Args:
            surface: Pygame 表面
        """
        self._ensure_fonts()
        
        # 绘制背景
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.bg_color, bg_rect)
        pygame.draw.rect(surface, (100, 100, 100), bg_rect, 2)  # 边框
        
        # 获取排名数据
        rankings = self.scoring_system.get_rankings()
        
        if not rankings:
            # 无数据提示
            text = self._font_normal.render("No agents registered", True, self.text_color)
            text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
            surface.blit(text, text_rect)
            return
        
        # 计算行高和列位置
        header_height = 35
        row_height = 30
        total_rows = min(len(rankings) + 1, (self.height - header_height) // row_height)
        
        # 绘制表头
        header_rect = pygame.Rect(self.x, self.y, self.width, header_height)
        pygame.draw.rect(surface, self.header_color, header_rect)
        
        current_x = self.x + 5
        for col_name, col_width in self.columns:
            text = self._font_header.render(col_name, True, (200, 200, 200))
            text_rect = text.get_rect(midleft=(current_x + 5, self.y + header_height // 2))
            surface.blit(text, text_rect)
            current_x += col_width
        
        # 绘制数据行
        current_y = self.y + header_height
        for i in range(min(len(rankings), total_rows - 1)):
            agent_id, window_score = rankings[i]
            info = self._get_agent_info(agent_id, i + 1)
            
            # 交替行颜色
            row_color = self.row_colors[i % 2]
            row_rect = pygame.Rect(self.x, current_y, self.width, row_height)
            pygame.draw.rect(surface, row_color, row_rect)
            
            # 绘制各列数据
            current_x = self.x + 5
            
            # Rank
            text = self._font_normal.render(str(i + 1), True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[0][1]
            
            # Agent ID
            text = self._font_normal.render(info.agent_id, True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[1][1]
            
            # Score (window average)
            text = self._font_normal.render(f"{info.score:.2f}", True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[2][1]
            
            # Games
            text = self._font_normal.render(str(info.games_played), True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[3][1]
            
            # W/D/L
            wdl_text = f"{info.wins}/{info.draws}/{info.losses}"
            text = self._font_normal.render(wdl_text, True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[4][1]
            
            # Win%
            text = self._font_normal.render(f"{info.win_rate:.0f}%", True, self.text_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            current_x += self.columns[5][1]
            
            # Role
            role_color = (100, 200, 100) if info.role == "Teacher" else (
                (200, 200, 100) if info.role == "Student" else self.text_color
            )
            text = self._font_normal.render(info.role, True, role_color)
            surface.blit(text, (current_x + 5, current_y + 3))
            
            current_y += row_height
    
    def get_clicked_agent(self, mouse_x: int, mouse_y: int) -> Optional[str]:
        """
        获取点击的智能体 ID
        
        Returns:
            智能体 ID 或 None
        """
        if mouse_x < self.x or mouse_x > self.x + self.width:
            return None
        if mouse_y < self.y or mouse_y > self.y + self.height:
            return None
        
        header_height = 35
        row_height = 30
        
        if mouse_y < self.y + header_height:
            return None  # 点击了表头
        
        row_index = (mouse_y - self.y - header_height) // row_height
        rankings = self.scoring_system.get_rankings()
        
        if 0 <= row_index < len(rankings):
            return rankings[row_index][0]
        
        return None
