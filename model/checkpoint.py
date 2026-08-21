"""
模型检查点管理 - 保存/加载/版本管理

功能：
1. 保存模型权重、优化器状态、训练元数据
2. 加载检查点（支持版本回滚）
3. 自动版本管理（保留最近 N 个检查点）
4. 最佳模型跟踪（基于评估分数）
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch


class CheckpointManager:
    """
    检查点管理器
    
    负责模型的保存、加载和版本管理
    """
    
    def __init__(self, checkpoint_dir: str, max_versions: int = 5):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_versions: 保留的最大版本数
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_versions = max_versions
        self.best_score = float('-inf')
        self.best_checkpoint_path: Optional[Path] = None
        
        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载元数据
        self.metadata = self._load_metadata()
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, score: float, extra_data: Dict[str, Any] = None) -> Path:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前轮次
            score: 当前分数
            extra_data: 额外数据
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_epoch{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': timestamp,
            'metadata': extra_data or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 更新元数据
        self._update_metadata(epoch, score, checkpoint_name)
        
        # 检查是否为最佳模型
        if score > self.best_score:
            self.best_score = score
            self.best_checkpoint_path = checkpoint_path
            self._save_best_checkpoint(checkpoint_path)
        
        # 清理旧版本
        self._cleanup_old_versions()
        
        return checkpoint_path
    
    def load(self, checkpoint_path: Optional[str] = None, 
             model: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径（None 则加载最新）
            model: 要加载权重的模型
            optimizer: 要加载状态的优化器
            
        Returns:
            检查点数据字典
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("未找到检查点文件")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型权重
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def load_best(self, model: torch.nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        加载最佳模型检查点
        
        Args:
            model: 要加载权重的模型
            optimizer: 要加载状态的优化器
            
        Returns:
            检查点数据字典
        """
        if self.best_checkpoint_path is None:
            # 尝试从元数据中恢复
            if 'best_checkpoint' in self.metadata:
                self.best_checkpoint_path = self.checkpoint_dir / self.metadata['best_checkpoint']
        
        if self.best_checkpoint_path is None or not self.best_checkpoint_path.exists():
            raise FileNotFoundError("未找到最佳模型检查点")
        
        return self.load(self.best_checkpoint_path, model, optimizer)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有检查点
        
        Returns:
            检查点信息列表
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            if path.name.startswith('best_'):
                continue
            
            try:
                checkpoint = torch.load(path, map_location='cpu')
                checkpoints.append({
                    'path': str(path),
                    'epoch': checkpoint.get('epoch', 0),
                    'score': checkpoint.get('score', 0.0),
                    'timestamp': checkpoint.get('timestamp', ''),
                    'size_mb': path.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                print(f"警告：无法读取检查点 {path}: {e}")
        
        # 按时间戳排序（最新的在前）
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        删除指定检查点
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            是否删除成功
        """
        path = Path(checkpoint_path)
        if path.exists():
            path.unlink()
            self._update_metadata_file()
            return True
        return False
    
    def _update_metadata(self, epoch: int, score: float, checkpoint_name: str):
        """更新元数据"""
        if 'checkpoints' not in self.metadata:
            self.metadata['checkpoints'] = []
        
        self.metadata['checkpoints'].append({
            'name': checkpoint_name,
            'epoch': epoch,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保留最近的记录
        if len(self.metadata['checkpoints']) > self.max_versions * 2:
            self.metadata['checkpoints'] = self.metadata['checkpoints'][-self.max_versions * 2:]
        
        self._save_metadata()
    
    def _save_best_checkpoint(self, source_path: Path):
        """保存最佳检查点副本"""
        best_path = self.checkpoint_dir / "best_model.pt"
        shutil.copy2(source_path, best_path)
        self.metadata['best_checkpoint'] = "best_model.pt"
        self.metadata['best_score'] = self.best_score
        self._save_metadata()
    
    def _cleanup_old_versions(self):
        """清理旧版本，只保留最近的 max_versions 个"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_versions:
            # 删除旧的检查点（保留最佳的）
            for cp in checkpoints[self.max_versions:]:
                cp_path = Path(cp['path'])
                if cp_path != self.best_checkpoint_path:
                    try:
                        cp_path.unlink()
                        print(f"已删除旧检查点：{cp_path.name}")
                    except Exception as e:
                        print(f"删除检查点失败：{e}")
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点"""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return Path(checkpoints[0]['path'])
        return None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_path = self.checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _update_metadata_file(self):
        """更新元数据文件（删除检查点后调用）"""
        # 重新扫描目录构建元数据
        valid_checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            if path.name.startswith('best_'):
                continue
            try:
                checkpoint = torch.load(path, map_location='cpu')
                valid_checkpoints.append({
                    'name': path.name,
                    'epoch': checkpoint.get('epoch', 0),
                    'score': checkpoint.get('score', 0.0),
                    'timestamp': checkpoint.get('timestamp', '')
                })
            except:
                pass
        
        self.metadata['checkpoints'] = valid_checkpoints
        self._save_metadata()


# 便捷函数
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, score: float, save_dir: str,
                    extra_data: Dict[str, Any] = None) -> Path:
    """
    便捷函数：保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 轮次
        score: 分数
        save_dir: 保存目录
        extra_data: 额外数据
        
    Returns:
        保存路径
    """
    manager = CheckpointManager(save_dir)
    return manager.save(model, optimizer, epoch, score, extra_data)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    便捷函数：加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        
    Returns:
        检查点数据
    """
    manager = CheckpointManager(os.path.dirname(checkpoint_path))
    return manager.load(checkpoint_path, model, optimizer)
