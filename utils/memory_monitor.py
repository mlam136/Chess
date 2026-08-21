"""
内存监控模块 - 实时追踪进程内存使用情况

用法:
    from utils.memory_monitor import MemoryMonitor
    
    monitor = MemoryMonitor()
    
    # 获取当前内存
    current = monitor.get_current_memory()
    
    # 记录快照
    monitor.snapshot("before_training")
    
    # 运行训练
    train()
    
    # 对比内存变化
    monitor.log_usage("after_training")
    
    # 获取峰值
    peak = monitor.get_peak_memory()
"""

import psutil
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """内存快照数据"""
    timestamp: float
    memory_mb: float
    label: str
    pid: int


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self, pid: Optional[int] = None):
        """
        初始化内存监控器
        
        Args:
            pid: 要监控的进程 ID，默认为当前进程
        """
        self.pid = pid or os.getpid()
        self.process = psutil.Process(self.pid)
        self.snapshots: List[MemorySnapshot] = []
        self.peak_memory = 0.0
        self._initial_memory = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """
        获取当前进程内存占用 (MB)
        
        Returns:
            当前内存使用量 (MB)
        """
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)  # 转换为 MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_peak_memory(self) -> float:
        """
        获取历史峰值内存 (MB)
        
        Returns:
            峰值内存使用量 (MB)
        """
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)
        return self.peak_memory
    
    def reset_peak(self):
        """重置峰值记录"""
        self.peak_memory = self.get_current_memory()
    
    def snapshot(self, label: str = "") -> MemorySnapshot:
        """
        创建内存快照
        
        Args:
            label: 快照标签
        
        Returns:
            MemorySnapshot 对象
        """
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)
        
        snap = MemorySnapshot(
            timestamp=time.time(),
            memory_mb=current,
            label=label,
            pid=self.pid
        )
        self.snapshots.append(snap)
        return snap
    
    def log_usage(self, label: str = "", log_diff: bool = True) -> float:
        """
        记录带标签的内存使用情况
        
        Args:
            label: 标签
            log_diff: 是否记录与上次快照的差值
        
        Returns:
            当前内存使用量 (MB)
        """
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)
        
        snap = self.snapshot(label)
        
        if log_diff and len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            diff = current - prev.memory_mb
            print(f"[Memory] {label}: {current:.2f} MB ({diff:+.2f} MB from '{prev.label}')")
        else:
            print(f"[Memory] {label}: {current:.2f} MB")
        
        return current
    
    def get_memory_history(self) -> List[Dict]:
        """
        获取内存使用历史
        
        Returns:
            包含时间戳和内存值的字典列表
        """
        return [
            {
                'timestamp': s.timestamp,
                'memory_mb': s.memory_mb,
                'label': s.label
            }
            for s in self.snapshots
        ]
    
    def check_memory_limit(self, limit_mb: float) -> bool:
        """
        检查是否超过内存限制
        
        Args:
            limit_mb: 内存限制 (MB)
        
        Returns:
            True 如果未超限，False 如果已超限
        """
        current = self.get_current_memory()
        if current > limit_mb:
            print(f"[WARNING] Memory usage {current:.2f} MB exceeds limit {limit_mb:.2f} MB")
            return False
        return True
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取内存统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        current = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current)
        
        stats = {
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'initial_mb': self._initial_memory,
            'growth_mb': current - self._initial_memory,
            'snapshot_count': len(self.snapshots)
        }
        
        if self.snapshots:
            memories = [s.memory_mb for s in self.snapshots]
            stats['avg_mb'] = sum(memories) / len(memories)
            stats['min_mb'] = min(memories)
        else:
            stats['avg_mb'] = current
            stats['min_mb'] = current
        
        return stats
    
    def print_stats(self):
        """打印内存统计信息"""
        stats = self.get_memory_stats()
        
        print("\n" + "=" * 50)
        print("MEMORY USAGE STATISTICS")
        print("=" * 50)
        print(f"Current:   {stats['current_mb']:>10.2f} MB")
        print(f"Peak:      {stats['peak_mb']:>10.2f} MB")
        print(f"Initial:   {stats['initial_mb']:>10.2f} MB")
        print(f"Growth:    {stats['growth_mb']:>10.2f} MB")
        print(f"Average:   {stats['avg_mb']:>10.2f} MB")
        print(f"Minimum:   {stats['min_mb']:>10.2f} MB")
        print(f"Snapshots: {stats['snapshot_count']:>10}")
        print("=" * 50 + "\n")


# 便捷函数
_default_monitor: Optional[MemoryMonitor] = None


def get_default_monitor() -> MemoryMonitor:
    """获取默认的全局内存监控器"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = MemoryMonitor()
    return _default_monitor


def log_memory(label: str = "") -> float:
    """使用默认监控器记录内存使用"""
    return get_default_monitor().log_usage(label)


def check_memory(limit_mb: float) -> bool:
    """使用默认监控器检查内存限制"""
    return get_default_monitor().check_memory_limit(limit_mb)


if __name__ == "__main__":
    # 示例用法
    print("Running memory monitor demo...")
    
    monitor = MemoryMonitor()
    
    print(f"Initial memory: {monitor.get_current_memory():.2f} MB")
    
    # 模拟内存分配
    data = []
    for i in range(5):
        data.append([0] * (1024 * 1024))  # 分配约 4MB
        monitor.log_usage(f"iteration_{i}")
    
    # 打印统计
    monitor.print_stats()
    
    # 检查限制
    if not monitor.check_memory_limit(100.0):
        print("Memory limit exceeded!")
