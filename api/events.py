"""
事件总线 - 发布/订阅模式
用于模块间通信
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """事件类型枚举"""
    GAME_STARTED = "game_started"
    GAME_ENDED = "game_ended"
    MOVE_MADE = "move_made"
    GAME_PAUSED = "game_paused"
    GAME_RESUMED = "game_resumed"
    AGENT_REGISTERED = "agent_registered"
    SCORE_UPDATED = "score_updated"
    ROLE_CHANGED = "role_changed"
    CONFIG_RELOADED = "config_reloaded"
    ERROR = "error"


@dataclass
class Event:
    """事件对象"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = 0.0
    
    def __post_init__(self):
        import time
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# 回调函数类型
EventHandler = Callable[[Event], None]


class EventBus:
    """
    事件总线
    实现发布/订阅模式，用于模块解耦
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls) -> 'EventBus':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._event_history: List[Event] = []
        self._max_history = 100
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 回调函数
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            handler: 回调函数
        """
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
    
    def publish(self, event_type: EventType, data: Optional[Dict[str, Any]] = None) -> None:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        event = Event(
            event_type=event_type,
            data=data or {},
        )
        
        # 保存到历史记录
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
        
        # 通知订阅者
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler for {event_type}: {e}")
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50
    ) -> List[Event]:
        """
        获取历史事件
        
        Args:
            event_type: 过滤的事件类型（None 表示全部）
            limit: 返回数量限制
            
        Returns:
            List[Event]: 事件列表
        """
        if event_type is None:
            return self._event_history[-limit:]
        
        filtered = [
            e for e in self._event_history
            if e.event_type == event_type
        ]
        return filtered[-limit:]
    
    def clear_history(self) -> None:
        """清空历史事件"""
        self._event_history.clear()
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# 全局事件总线实例
event_bus = EventBus()


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    return event_bus
