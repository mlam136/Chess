"""
实时 Loss 曲线渲染
使用 Pygame 绘制训练过程中的 loss 变化曲线
支持多条曲线对比（total_loss, distill_loss, selfplay_loss）
"""

import pygame
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class LossDataPoint:
    """Loss 数据点"""
    step: int
    total_loss: float
    distill_loss: float = 0.0
    selfplay_loss: float = 0.0
    reg_loss: float = 0.0
    timestamp: float = field(default_factory=time.time)


class LossChart:
    """
    Loss 曲线图表组件
    实时显示训练过程中的 loss 变化趋势
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_points: int = 200,
    ):
        """
        初始化 Loss 图表

        Args:
            x: 左上角 X 坐标
            y: 左上角 Y 坐标
            width: 图表宽度
            height: 图表高度
            max_points: 最大显示数据点数
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points

        # 数据存储
        self._data_points: deque[LossDataPoint] = deque(maxlen=max_points)
        self._total_loss_history: deque[float] = deque(maxlen=max_points)
        self._distill_loss_history: deque[float] = deque(maxlen=max_points)
        self._selfplay_loss_history: deque[float] = deque(maxlen=max_points)

        # 样式配置
        self.bg_color = (35, 35, 45)
        self.grid_color = (50, 50, 60)
        self.text_color = (200, 200, 200)
        self.border_color = (100, 100, 120)

        # 曲线颜色
        self.colors = {
            'total': (255, 100, 100),      # 红色
            'distill': (100, 200, 255),    # 蓝色
            'selfplay': (100, 255, 150),   # 绿色
            'reg': (255, 200, 100),        # 黄色
        }

        # 字体
        self._font_title: Optional[pygame.font.Font] = None
        self._font_label: Optional[pygame.font.Font] = None
        self._font_value: Optional[pygame.font.Font] = None

        # Y 轴范围
        self._y_min = 0.0
        self._y_max = 5.0
        self._auto_scale = True

        # 显示选项
        self._show_total = True
        self._show_distill = True
        self._show_selfplay = True
        self._show_reg = False

    def _ensure_fonts(self) -> None:
        """确保字体已初始化"""
        if self._font_title is None:
            self._font_title = pygame.font.Font(None, 28)
            self._font_label = pygame.font.Font(None, 22)
            self._font_value = pygame.font.Font(None, 20)

    def add_data_point(
        self,
        step: int,
        total_loss: float,
        distill_loss: float = 0.0,
        selfplay_loss: float = 0.0,
        reg_loss: float = 0.0,
    ) -> None:
        """
        添加新的数据点

        Args:
            step: 训练步数
            total_loss: 总 loss
            distill_loss: 蒸馏 loss
            selfplay_loss: 自博弈 loss
            reg_loss: 正则化 loss
        """
        point = LossDataPoint(
            step=step,
            total_loss=total_loss,
            distill_loss=distill_loss,
            selfplay_loss=selfplay_loss,
            reg_loss=reg_loss,
        )
        self._data_points.append(point)
        self._total_loss_history.append(total_loss)
        self._distill_loss_history.append(distill_loss)
        self._selfplay_loss_history.append(selfplay_loss)

        # 自动调整 Y 轴范围
        if self._auto_scale and len(self._data_points) > 5:
            all_values = [p.total_loss for p in self._data_points]
            self._y_max = max(all_values) * 1.2 + 0.1
            self._y_min = min(0, min(all_values) * 0.8)

    def set_y_range(self, y_min: float, y_max: float) -> None:
        """手动设置 Y 轴范围"""
        self._y_min = y_min
        self._y_max = y_max
        self._auto_scale = False

    def toggle_curve(self, curve_name: str) -> None:
        """切换曲线显示状态"""
        if curve_name == 'total':
            self._show_total = not self._show_total
        elif curve_name == 'distill':
            self._show_distill = not self._show_distill
        elif curve_name == 'selfplay':
            self._show_selfplay = not self._show_selfplay
        elif curve_name == 'reg':
            self._show_reg = not self._show_reg

    def clear(self) -> None:
        """清空所有数据"""
        self._data_points.clear()
        self._total_loss_history.clear()
        self._distill_loss_history.clear()
        self._selfplay_loss_history.clear()
        self._y_min = 0.0
        self._y_max = 5.0

    def _get_legend_rects(self) -> List[Tuple[pygame.Rect, str]]:
        """获取图例区域"""
        legend_items = []
        start_x = self.x + 10
        start_y = self.y + self.height - 80

        if self._show_total:
            rect = pygame.Rect(start_x, start_y, 100, 20)
            legend_items.append((rect, 'Total'))
            start_y += 20

        if self._show_distill:
            rect = pygame.Rect(start_x, start_y, 100, 20)
            legend_items.append((rect, 'Distill'))
            start_y += 20

        if self._show_selfplay:
            rect = pygame.Rect(start_x, start_y, 100, 20)
            legend_items.append((rect, 'SelfPlay'))
            start_y += 20

        if self._show_reg:
            rect = pygame.Rect(start_x, start_y, 100, 20)
            legend_items.append((rect, 'Reg'))

        return legend_items

    def _draw_grid(self, surface: pygame.Surface) -> None:
        """绘制网格背景"""
        # 水平网格线
        num_lines = 5
        chart_height = self.height - 60  # 留出标题和图例空间
        step_y = chart_height / num_lines

        for i in range(num_lines + 1):
            y = self.y + 30 + i * step_y
            pygame.draw.line(surface, self.grid_color, (self.x + 50, y), 
                           (self.x + self.width - 10, y), 1)
            
            # Y 轴标签
            value = self._y_max - (self._y_max - self._y_min) * i / num_lines
            label = self._font_label.render(f"{value:.2f}", True, self.text_color)
            surface.blit(label, (self.x + 5, y - 10))

        # 垂直网格线
        num_vlines = 5
        chart_width = self.width - 70
        step_x = chart_width / num_vlines

        for i in range(num_vlines + 1):
            x = self.x + 60 + i * step_x
            pygame.draw.line(surface, self.grid_color, (x, self.y + 30),
                           (x, self.y + self.height - 50), 1)

    def _draw_curve(
        self,
        surface: pygame.Surface,
        values: List[float],
        color: Tuple[int, int, int],
        line_width: int = 2,
    ) -> None:
        """绘制单条曲线"""
        if len(values) < 2:
            return

        chart_width = self.width - 70
        chart_height = self.height - 80
        start_x = self.x + 60
        start_y = self.y + 30

        points = []
        for i, value in enumerate(values):
            # 归一化到图表区域
            x = start_x + (i / max(len(values) - 1, 1)) * chart_width
            y_range = self._y_max - self._y_min
            if y_range > 0:
                normalized_y = (value - self._y_min) / y_range
            else:
                normalized_y = 0.5
            y = start_y + chart_height * (1 - normalized_y)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, line_width)

    def draw(self, surface: pygame.Surface) -> None:
        """
        绘制图表

        Args:
            surface: Pygame 表面
        """
        self._ensure_fonts()

        # 绘制背景
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.bg_color, bg_rect)
        pygame.draw.rect(surface, self.border_color, bg_rect, 2)

        # 绘制标题
        title = self._font_title.render("Loss Curve", True, self.text_color)
        surface.blit(title, (self.x + 10, self.y + 5))

        # 绘制网格
        self._draw_grid(surface)

        # 绘制曲线
        if self._show_total and len(self._total_loss_history) > 0:
            self._draw_curve(surface, list(self._total_loss_history), 
                           self.colors['total'], 3)

        if self._show_distill and len(self._distill_loss_history) > 0:
            self._draw_curve(surface, list(self._distill_loss_history),
                           self.colors['distill'], 2)

        if self._show_selfplay and len(self._selfplay_loss_history) > 0:
            self._draw_curve(surface, list(self._selfplay_loss_history),
                           self.colors['selfplay'], 2)

        # 绘制图例
        self._draw_legend(surface)

        # 绘制当前值
        if len(self._data_points) > 0:
            self._draw_current_values(surface)

    def _draw_legend(self, surface: pygame.Surface) -> None:
        """绘制图例"""
        legend_x = self.x + 10
        legend_y = self.y + self.height - 70

        items = [
            ('Total', self.colors['total'], self._show_total),
            ('Distill', self.colors['distill'], self._show_distill),
            ('SelfPlay', self.colors['selfplay'], self._show_selfplay),
            ('Reg', self.colors['reg'], self._show_reg),
        ]

        for i, (name, color, show) in enumerate(items):
            if not show:
                continue
            y = legend_y + i * 18
            
            # 颜色样本
            sample_rect = pygame.Rect(legend_x, y, 15, 15)
            pygame.draw.rect(surface, color, sample_rect)
            pygame.draw.rect(surface, (150, 150, 150), sample_rect, 1)
            
            # 标签
            label = self._font_label.render(name, True, self.text_color)
            surface.blit(label, (legend_x + 20, y))

    def _draw_current_values(self, surface: pygame.Surface) -> None:
        """绘制当前数值"""
        latest = self._data_points[-1]
        value_x = self.x + self.width - 150
        value_y = self.y + 10

        text = self._font_value.render(f"Step: {latest.step}", True, self.text_color)
        surface.blit(text, (value_x, value_y))

        if self._show_total:
            text = self._font_value.render(
                f"Total: {latest.total_loss:.4f}",
                True, self.colors['total']
            )
            surface.blit(text, (value_x, value_y + 18))

    def get_clicked_curve(self, mouse_x: int, mouse_y: int) -> Optional[str]:
        """
        检测点击了哪条曲线（用于切换显示）

        Returns:
            曲线名称或 None
        """
        legend_items = self._get_legend_rects()
        
        for rect, name in legend_items:
            if rect.collidepoint(mouse_x, mouse_y):
                return name
        
        return None


class MultiChartPanel:
    """
    多图表面板
    可并排显示多个 Loss 图表（用于 α/β消融对比）
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        num_charts: int = 1,
        titles: List[str] = None,
    ):
        """
        初始化多图表面板

        Args:
            x: 左上角 X 坐标
            y: 左上角 Y 坐标
            width: 面板总宽度
            height: 面板总高度
            num_charts: 图表数量
            titles: 各图表标题列表
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.num_charts = num_charts

        # 创建子图表
        chart_width = (width - 20 * (num_charts - 1)) // num_charts
        self._charts: List[LossChart] = []

        titles = titles or [f"Chart {i+1}" for i in range(num_charts)]

        for i in range(num_charts):
            chart_x = x + i * (chart_width + 20)
            chart = LossChart(
                x=chart_x,
                y=y,
                width=chart_width,
                height=height,
            )
            self._charts.append(chart)

        self._titles = titles[:num_charts]

    def add_data_point(
        self,
        chart_index: int,
        step: int,
        total_loss: float,
        distill_loss: float = 0.0,
        selfplay_loss: float = 0.0,
        reg_loss: float = 0.0,
    ) -> None:
        """向指定图表添加数据点"""
        if 0 <= chart_index < len(self._charts):
            self._charts[chart_index].add_data_point(
                step, total_loss, distill_loss, selfplay_loss, reg_loss
            )

    def clear_all(self) -> None:
        """清空所有图表"""
        for chart in self._charts:
            chart.clear()

    def draw(self, surface: pygame.Surface) -> None:
        """绘制所有图表"""
        for chart in self._charts:
            chart.draw(surface)

    def get_chart_at_position(
        self, mouse_x: int, mouse_y: int
    ) -> Optional[Tuple[int, LossChart]]:
        """
        获取指定位置的图表

        Returns:
            (chart_index, chart) 或 None
        """
        for i, chart in enumerate(self._charts):
            if (chart.x <= mouse_x <= chart.x + chart.width and
                chart.y <= mouse_y <= chart.y + chart.height):
                return (i, chart)
        return None
