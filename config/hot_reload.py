"""
配置热更新模块
运行时动态加载和更新配置文件，无需重启
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """全局配置管理器，支持热更新"""
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    _config_path: Optional[Path] = None
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
    
    def load(self, config_path: str = "config/default.yaml") -> None:
        """加载配置文件"""
        self._config_path = Path(config_path)
        self._reload()
    
    def _reload(self) -> None:
        """重新加载配置文件"""
        if self._config_path and self._config_path.exists():
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
    
    def reload(self) -> None:
        """公开的重载方法，用于运行时热更新"""
        self._reload()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键（如 'VISUALIZATION.board_size'）"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值（仅内存中，不写回文件）"""
        keys = key.split('.')
        config = self._config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getattr__(self, name: str) -> Any:
        """支持直接属性访问"""
        if name in self._config_data:
            return self._config_data[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    @property
    def data(self) -> Dict[str, Any]:
        """返回完整配置数据"""
        return self._config_data.copy()


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置实例"""
    return config


def reload_config() -> None:
    """重新加载配置（热更新）"""
    config.reload()
