"""
性能分析工具 - 用于测量代码块执行时间和系统资源使用

用法:
    from utils.profiler import ProfileContext, print_profile_report
    
    with ProfileContext("mcts_search"):
        result = mcts_search(...)
    
    print_profile_report()
"""

import time
import threading
from contextlib import contextmanager
from typing import Dict, List
from collections import defaultdict


class ProfileData:
    """存储单个函数的性能数据"""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
    
    def record(self, duration: float):
        """记录一次调用耗时"""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
    
    @property
    def avg_time(self) -> float:
        """平均耗时"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0


# 全局性能数据存储
_profile_data: Dict[str, ProfileData] = {}
_lock = threading.Lock()


def _get_or_create_data(name: str) -> ProfileData:
    """获取或创建性能数据对象"""
    if name not in _profile_data:
        _profile_data[name] = ProfileData(name)
    return _profile_data[name]


class ProfileContext:
    """上下文管理器，用于测量代码块执行时间"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self._thread_data = threading.local()
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        
        with _lock:
            data = _get_or_create_data(self.name)
            data.record(duration)
        
        return False  # 不抑制异常


@contextmanager
def profile_block(name: str):
    """
    装饰器风格的性能分析上下文
    
    用法:
        with profile_block("my_function"):
            # 要测量的代码
            pass
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        duration = end - start
        
        with _lock:
            data = _get_or_create_data(name)
            data.record(duration)


def get_profile_data() -> Dict[str, ProfileData]:
    """获取所有性能数据"""
    with _lock:
        return dict(_profile_data)


def clear_profile_data():
    """清除所有性能数据"""
    global _profile_data
    with _lock:
        _profile_data.clear()


def print_profile_report(sort_by: str = 'total'):
    """
    打印性能分析报告
    
    Args:
        sort_by: 排序方式 ('total', 'avg', 'count', 'name')
    """
    with _lock:
        if not _profile_data:
            print("No profile data recorded.")
            return
        
        data_list = list(_profile_data.values())
        
        # 排序
        if sort_by == 'total':
            data_list.sort(key=lambda x: x.total_time, reverse=True)
        elif sort_by == 'avg':
            data_list.sort(key=lambda x: x.avg_time, reverse=True)
        elif sort_by == 'count':
            data_list.sort(key=lambda x: x.call_count, reverse=True)
        elif sort_by == 'name':
            data_list.sort(key=lambda x: x.name)
        
        # 打印报告
        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILE REPORT")
        print("=" * 80)
        print(f"{'Function':<40} {'Calls':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        print("-" * 80)
        
        for data in data_list:
            print(
                f"{data.name:<40} "
                f"{data.call_count:>8} "
                f"{data.total_time:>12.4f} "
                f"{data.avg_time * 1000:>12.3f} "
                f"{data.min_time * 1000:>12.3f} "
                f"{data.max_time * 1000:>12.3f}"
            )
        
        print("=" * 80 + "\n")


def benchmark(func, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
    """
    基准测试函数
    
    Args:
        func: 要测试的函数
        *args: 位置参数
        iterations: 迭代次数
        **kwargs: 关键字参数
    
    Returns:
        包含统计信息的字典
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'total': sum(times)
    }


if __name__ == "__main__":
    # 示例用法
    print("Running profiler demo...")
    
    with ProfileContext("example_function"):
        time.sleep(0.1)
    
    with ProfileContext("another_function"):
        time.sleep(0.05)
    
    # 嵌套示例
    with ProfileContext("outer"):
        with ProfileContext("inner"):
            time.sleep(0.02)
    
    print_profile_report()
