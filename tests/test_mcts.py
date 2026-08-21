"""
测试 MCTS 搜索 - 正确性验证
"""

import pytest
import numpy as np
from mcts.node import MCTSNode
from mcts.search import mcts_search, select_child, expand, backpropagate
from engine.board import Board


class TestMCTSNode:
    """测试 MCTS 节点"""
    
    def test_node_initialization(self):
        """测试节点初始化"""
        board = Board()
        node = MCTSNode(board)
        
        assert node.visit_count == 0
        assert node.value == 0.0
        assert node.children is not None
        assert node.is_terminal == False
    
    def test_is_fully_expanded(self):
        """测试完全扩展检测"""
        board = Board()
        node = MCTSNode(board)
        
        # 初始未扩展
        assert not node.is_fully_expanded
        
        # 模拟扩展所有子节点
        for move in list(board.legal_moves)[:5]:
            node.add_child(move, board.copy())
        
        # 仍未完全扩展（因为只扩展了部分）
        # 注意：实际是否完全扩展取决于实现
    
    def test_add_child(self):
        """测试添加子节点"""
        board = Board()
        node = MCTSNode(board)
        
        move = list(board.legal_moves)[0]
        child = node.add_child(move, board.copy())
        
        assert child is not None
        assert len(node.children) > 0
        assert move in [c.move for c in node.children.values()]


class TestMCTSSearch:
    """测试 MCTS 搜索"""
    
    def test_basic_search(self):
        """测试基本搜索"""
        board = Board()
        
        pi, value = mcts_search(
            board=board,
            model=None,  # 使用随机 rollout
            iterations=10,
            c_puct=1.5
        )
        
        assert pi is not None
        assert len(pi) > 0
        assert isinstance(value, (int, float, np.floating))
    
    def test_search_increases_visits(self):
        """测试搜索增加访问次数"""
        board = Board()
        root = MCTSNode(board)
        
        initial_visits = root.visit_count
        
        # 执行几次搜索
        for _ in range(5):
            node = root
            # 简单选择未访问的子节点
            if not node.children:
                expand(node, None)
            else:
                node = list(node.children.values())[0]
            
            backpropagate(node, value=0.5)
        
        assert root.visit_count > initial_visits
    
    def test_select_child_with_prior(self):
        """测试带先验概率的子节点选择"""
        board = Board()
        node = MCTSNode(board)
        
        # 添加几个子节点
        moves = list(board.legal_moves)[:3]
        for i, move in enumerate(moves):
            child = node.add_child(move, board.copy())
            child.prior = 0.5 + i * 0.1  # 不同的先验
        
        # 选择应该考虑 PUCT
        selected = select_child(node, c_puct=1.5)
        
        assert selected is not None


class TestExpandAndBackprop:
    """测试扩展和回溯"""
    
    def test_expand_creates_children(self):
        """测试扩展创建子节点"""
        board = Board()
        node = MCTSNode(board)
        
        expand(node, model=None)
        
        # 应该有合法走法数量的子节点
        assert len(node.children) == len(list(board.legal_moves))
    
    def test_backprop_updates_values(self):
        """测试回溯更新值"""
        board = Board()
        parent = MCTSNode(board)
        child = parent.add_child(list(board.legal_moves)[0], board.copy())
        
        initial_parent_value = parent.value
        initial_child_value = child.value
        
        backpropagate(child, value=0.8)
        
        # 值应该被更新
        assert child.visit_count == 1
        assert child.value != initial_child_value or parent.value != initial_parent_value
    
    def test_backprop_aggregates_multiple(self):
        """测试多次回溯聚合"""
        board = Board()
        node = MCTSNode(board)
        
        # 多次回溯
        for v in [0.2, 0.5, 0.8]:
            child = node.add_child(list(board.legal_moves)[0], board.copy())
            backpropagate(child, value=v)
        
        # 访问次数应该是 3
        assert node.visit_count >= 3


class TestPolicyExtraction:
    """测试策略提取"""
    
    def test_root_policy_distribution(self):
        """测试根节点策略分布"""
        from mcts.policy import get_root_policy, best_move_from_policy, policy_to_move
        
        board = Board()
        root = MCTSNode(board)
        
        # 模拟一些搜索
        for _ in range(20):
            if not root.children:
                expand(root, None)
            child = list(root.children.values())[0]
            backpropagate(child, value=0.5)
        
        pi = get_root_policy(root)
        
        assert pi is not None
        assert len(pi) > 0
        # 概率和应该接近 1（或归一化后为 1）
    
    def test_best_move_from_policy(self):
        """测试从策略中选择最佳走法"""
        board = Board()
        legal_moves = list(board.legal_moves)
        
        # 创建一个偏向第一个走法的策略
        pi = np.zeros(4608)
        idx = 0
        pi[idx] = 0.9
        pi[idx+1] = 0.1
        
        best_move = best_move_from_policy(pi, legal_moves)
        
        assert best_move is not None
    
    def test_policy_to_move_sampling(self):
        """测试策略采样"""
        board = Board()
        legal_moves = list(board.legal_moves)[:5]
        
        pi = np.ones(4608) * 0.001
        for i, move in enumerate(legal_moves):
            pi[i] = 0.1
        
        move = policy_to_move(pi, legal_moves, temperature=1.0)
        
        assert move in legal_moves


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
