"""
日志面板 - 实时显示系统日志和事件
"""

import pygame
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    message: str
    level: str = "INFO"  # INFO, WARNING, ERROR, DEBUG
    color: Tuple[int, int, int] = (255, 255, 255)  # 白色


class LogPanel:
    """
    日志面板组件
    显示实时系统日志和事件
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_lines: int = 20,
        font_size: int = 18,
    ):
        """
        初始化日志面板
        
        Args:
            x: X 坐标
            y: Y 坐标
            width: 宽度
            height: 高度
            max_lines: 最大显示行数
            font_size: 字体大小
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_lines = max_lines
        self.font_size = font_size
        
        # 日志列表
        self._logs: List[LogEntry] = []
        
        # 字体
        self.font = pygame.font.Font(None, font_size)
        self.small_font = pygame.font.Font(None, font_size - 2)
        
        # 背景颜色
        self.bg_color = (25, 25, 40)
        self.border_color = (60, 60, 80)
        
        # 滚动偏移
        self._scroll_offset = 0
    
    def add_log(
        self,
        message: str,
        level: str = "INFO",
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        添加日志条目
        
        Args:
            message: 日志消息
            level: 日志级别
            color: 文字颜色（None 则使用默认）
        """
        # 设置颜色
        if color is None:
            if level == "ERROR":
                color = (255, 100, 100)  # 红色
            elif level == "WARNING":
                color = (255, 200, 100)  # 橙色
            elif level == "DEBUG":
                color = (150, 150, 150)  # 灰色
            else:
                color = (255, 255, 255)  # 白色
        
        entry = LogEntry(
            timestamp=time.time(),
            message=message,
            level=level,
            color=color,
        )
        
        self._logs.append(entry)
        
        # 限制最大行数
        if len(self._logs) > self.max_lines * 2:  # 保留双倍缓冲用于滚动
            self._logs = self._logs[-self.max_lines * 2:]
    
    def clear(self) -> None:
        """清空日志"""
        self._logs.clear()
        self._scroll_offset = 0
    
    def _get_visible_logs(self) -> List[LogEntry]:
        """获取当前可见的日志"""
        start = max(0, len(self._logs) - self.max_lines - self._scroll_offset)
        end = max(0, len(self._logs) - self._scroll_offset)
        return self._logs[start:end]
    
    def handle_scroll(self, delta: int) -> None:
        """
        处理滚动事件
        
        Args:
            delta: 滚动增量（正数向上，负数向下）
        """
        max_scroll = max(0, len(self._logs) - self.max_lines)
        self._scroll_offset = max(0, min(max_scroll, self._scroll_offset + delta))
    
    def draw(self, screen: pygame.Surface) -> None:
        """绘制日志面板"""
        # 背景
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, bg_rect)
        pygame.draw.rect(screen, self.border_color, bg_rect, 2)
        
        # 标题
        title = self.font.render("Event Log", True, (200, 200, 220))
        screen.blit(title, (self.x + 10, self.y + 8))
        
        # 日志内容
        visible_logs = self._get_visible_logs()
        line_height = self.font_size + 4
        start_y = self.y + 35
        
        for i, log in enumerate(visible_logs):
            y = start_y + i * line_height
            
            # 时间戳
            time_str = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
            time_surf = self.small_font.render(time_str, True, (120, 120, 140))
            screen.blit(time_surf, (self.x + 10, y))
            
            # 日志级别
            level_surf = self.small_font.render(f"[{log.level}]", True, log.color)
            level_width = level_surf.get_width()
            screen.blit(level_surf, (self.x + 70, y))
            
            # 消息
            msg_surf = self.font.render(log.message, True, log.color)
            screen.blit(msg_surf, (self.x + 70 + level_width + 5, y))
        
        # 滚动指示器
        if len(self._logs) > self.max_lines:
            indicator_y = self.y + self.height - 20
            indicator = self.small_font.render(
                f"{len(self._logs) - self._scroll_offset - len(visible_logs)}/{len(self._logs)}",
                True, (100, 100, 120),
            )
            screen.blit(indicator, (self.x + self.width - 60, indicator_y))
    
    def info(self, message: str) -> None:
        """添加 INFO 级别日志"""
        self.add_log(message, "INFO")
    
    def warning(self, message: str) -> None:
        """添加 WARNING 级别日志"""
        self.add_log(message, "WARNING")
    
    def error(self, message: str) -> None:
        """添加 ERROR 级别日志"""
        self.add_log(message, "ERROR")
    
    def debug(self, message: str) -> None:
        """添加 DEBUG 级别日志"""
        self.add_log(message, "DEBUG")
