"""
训练模式的可视化覆盖层
显示实时对局、训练指标和控制面板

注意：需要 tkinter 支持，如无 GUI 环境将自动降级为日志模式
"""
import threading
import queue
from typing import Optional, Dict, Any
from datetime import datetime

# 延迟导入 tkinter，以便在无 GUI 环境中 gracefully degrade
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None
    ttk = None

from .board_widget import BoardWidget
from .loss_chart import LossChart
from .log_panel import LogPanel


class TrainingOverlay:
    """训练过程中的可视化监控面板"""
    
    def __init__(self, root: tk.Tk, update_queue: queue.Queue):
        if not TKINTER_AVAILABLE:
            raise ImportError("tkinter 不可用，无法创建 GUI 窗口")
        
        self.root = root
        self.update_queue = update_queue
        self.root.title("AlphaZero Chess - 训练监控")
        
        # 状态变量
        self.current_epoch = 0
        self.current_game = 0
        self.current_loss = 0.0
        self.is_training = False
        
        self._setup_ui()
        self._start_update_loop()
        
    def _setup_ui(self):
        """构建界面布局"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 顶部控制栏
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(control_frame, text="状态: 就绪", foreground="green")
        self.status_label.grid(row=0, column=0, padx=10)
        
        self.epoch_label = ttk.Label(control_frame, text="Epoch: 0/0")
        self.epoch_label.grid(row=0, column=1, padx=10)
        
        self.loss_label = ttk.Label(control_frame, text="Loss: 0.0000")
        self.loss_label.grid(row=0, column=2, padx=10)
        
        self.stop_btn = ttk.Button(control_frame, text="停止训练", command=self._request_stop)
        self.stop_btn.grid(row=0, column=3, padx=10)
        self.stop_btn.config(state="disabled")
        
        # 左侧：实时棋盘
        board_frame = ttk.LabelFrame(main_frame, text="实时对局", padding="5")
        board_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 5))
        
        self.board_widget = BoardWidget(board_frame, size=480)
        self.board_widget.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：图表和日志
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Loss 曲线图
        chart_frame = ttk.LabelFrame(right_frame, text="Loss 趋势", padding="5")
        chart_frame.pack(fill=tk.X, pady=(0, 5))
        self.loss_chart = LossChart(chart_frame, width=400, height=200)
        self.loss_chart.pack(fill=tk.BOTH, expand=True)
        
        # 日志面板
        log_frame = ttk.LabelFrame(right_frame, text="训练日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_panel = LogPanel(log_frame, height=15)
        self.log_panel.pack(fill=tk.BOTH, expand=True)
        
        # 行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def _start_update_loop(self):
        """启动 UI 更新循环"""
        self._process_queue()
        
    def _process_queue(self):
        """处理来自训练线程的消息队列"""
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                
                if msg_type == "status":
                    self._update_status(data)
                elif msg_type == "board":
                    self._update_board(data)
                elif msg_type == "metrics":
                    self._update_metrics(data)
                elif msg_type == "log":
                    self.log_panel.append(data)
                elif msg_type == "stop":
                    self._on_training_finished()
                    
        except queue.Empty:
            pass
        
        # 每 100ms 检查一次队列
        self.root.after(100, self._process_queue)
        
    def _update_status(self, data: Dict[str, Any]):
        """更新状态栏"""
        if data.get("training"):
            self.is_training = True
            self.status_label.config(text="状态: 训练中", foreground="blue")
            self.stop_btn.config(state="normal")
            self.epoch_label.config(text=f"Epoch: {data.get('epoch', 0)}/{data.get('total_epochs', 0)}")
        else:
            self.is_training = False
            self.status_label.config(text="状态: 空闲", foreground="green")
            self.stop_btn.config(state="disabled")
            
    def _update_board(self, board_state):
        """更新棋盘显示"""
        self.board_widget.update_board(board_state)
        
    def _update_metrics(self, data: Dict[str, Any]):
        """更新指标和图表"""
        if "loss" in data:
            loss = data["loss"]
            self.current_loss = loss
            self.loss_label.config(text=f"Loss: {loss:.4f}")
            self.loss_chart.add_point(loss)
            
        if "epoch" in data and "total_epochs" in data:
            self.epoch_label.config(text=f"Epoch: {data['epoch']}/{data['total_epochs']}")
            
    def _request_stop(self):
        """请求停止训练"""
        self.update_queue.put(("stop_request", {}))
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="状态: 正在停止...", foreground="orange")
        
    def _on_training_finished(self):
        """训练结束处理"""
        self.is_training = False
        self.status_label.config(text="状态: 训练完成", foreground="green")
        self.log_panel.append("=== 训练任务已完成 ===")
        
    def start_training_session(self, total_epochs: int):
        """初始化训练会话"""
        self.log_panel.append(f"=== 开始训练会话 | 总 Epochs: {total_epochs} | 时间: {datetime.now().strftime('%H:%M:%S')} ===")
        self.loss_chart.clear()
        self._update_status({"training": True, "total_epochs": total_epochs})
