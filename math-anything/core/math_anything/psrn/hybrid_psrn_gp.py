"""Hybrid PSRN-GP - 混合架构结合 PSRN 的快速探索与 GP 的精细优化.

核心思想：
Phase 1 (PSRN): 快速并行筛选，生成高质量初始候选
Phase 2 (GP): 基于候选进行遗传进化，精细优化

优势：
- 比纯 PSRN 更准确
- 比纯 GP 更快收敛
- 结合两者优点
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import warnings

from .psrn_network import PSRN, PSRNConfig
from ..eml_v2 import ImprovedSymbolicRegression, Node


@dataclass
class HybridConfig:
    """混合架构配置."""
    
    # PSRN 阶段配置
    psrn_layers: int = 2
    psrn_top_k: int = 50  # PSRN 阶段保留的候选数
    psrn_token_generator: str = "fast"
    
    # GP 阶段配置
    gp_population_size: int = 100
    gp_generations: int = 30
    gp_use_eml: bool = True
    
    # 混合策略
    use_psrn_seeding: bool = True  # 是否用 PSRN 结果初始化 GP
    diversity_boost: float = 0.3   # 多样性增强比例


class HybridPSRNGP:
    """PSRN-GP 混合符号回归."""
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        
        # 初始化 PSRN
        psrn_config = PSRNConfig(
            n_layers=self.config.psrn_layers,
            n_input_slots=5,
        )
        self.psrn = PSRN(psrn_config)
        
        # GP 将在 fit 时初始化（需要知道变量数）
        self.gp: Optional[ImprovedSymbolicRegression] = None
        
        # 存储结果
        self.psrn_candidates: List[Tuple[str, float]] = []
        self.best_tree: Optional[Node] = None
        self.best_fitness: float = float('inf')
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> Node:
        """运行混合搜索.
        
        Returns:
            best_tree: 最优表达式树
        """
        print("=" * 60)
        print("Hybrid PSRN-GP Search")
        print("=" * 60)
        
        # Phase 1: PSRN 快速探索
        print("\nPhase 1: PSRN Exploration")
        print("-" * 40)
        
        import time
        t0 = time.time()
        
        # 使用 PSRNSymbolicRegression API
        from .bridge import PSRNSymbolicRegression
        
        psrn_sr = PSRNSymbolicRegression(
            n_layers=self.config.psrn_layers,
            max_iterations=3,
            token_generator=self.config.psrn_token_generator,
        )
        
        psrn_sr.fit(X, y, variable_names)
        psrn_time = time.time() - t0
        
        # 获取 PSRN 候选
        self.psrn_candidates = self._extract_psrn_candidates(psrn_sr, X, y, variable_names)
        
        print(f"  PSRN found {len(self.psrn_candidates)} candidates")
        print(f"  Best PSRN MSE: {self.psrn_candidates[0][1]:.6f}")
        print(f"  Time: {psrn_time:.2f}s")
        
        # Phase 2: GP 精细优化
        print("\nPhase 2: GP Refinement")
        print("-" * 40)
        
        t0 = time.time()
        
        # 初始化 GP
        self.gp = ImprovedSymbolicRegression(
            population_size=self.config.gp_population_size,
            generations=self.config.gp_generations,
        )
        
        # 用 PSRN 候选初始化种群
        if self.config.use_psrn_seeding and self.psrn_candidates:
            initial_population = self._build_initial_population(
                self.psrn_candidates, X, y, variable_names
            )
            self.gp.population = initial_population
        
        # 运行 GP
        self.best_tree = self.gp.fit(X, y, variable_names)
        self.best_fitness = self.gp.best_fitness_
        
        gp_time = time.time() - t0
        
        print(f"  Best GP MSE: {self.best_fitness:.6f}")
        print(f"  Time: {gp_time:.2f}s")
        
        # 总结果
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  PSRN Phase:  {psrn_time:.2f}s, best MSE: {self.psrn_candidates[0][1]:.6f}")
        print(f"  GP Phase:    {gp_time:.2f}s, best MSE: {self.best_fitness:.6f}")
        print(f"  Total:       {psrn_time + gp_time:.2f}s")
        print(f"  Improvement: {(self.psrn_candidates[0][1] - self.best_fitness):.6f}")
        
        return self.best_tree
    
    def _extract_psrn_candidates(
        self,
        psrn_sr: 'PSRNSymbolicRegression',
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """从 PSRN 结果中提取候选."""
        from .compiled_evaluator import CompiledEvaluator
        evaluator = CompiledEvaluator()
        
        candidates = []
        
        # 获取最佳表达式
        best_expr = psrn_sr._best_expr
        if best_expr:
            try:
                y_pred = evaluator.evaluate(best_expr, X, variable_names or ['x'])
                mse = np.mean((y_pred - y) ** 2)
                candidates.append((best_expr, mse))
            except:
                pass
        
        # 尝试获取更多候选（通过不同配置运行多次）
        for token_gen in ['fast', 'random']:
            try:
                sr = PSRNSymbolicRegression(
                    n_layers=self.config.psrn_layers,
                    max_iterations=2,
                    token_generator=token_gen,
                )
                sr.fit(X, y, variable_names)
                
                expr = sr._best_expr
                if expr and expr not in [c[0] for c in candidates]:
                    y_pred = evaluator.evaluate(expr, X, variable_names or ['x'])
                    mse = np.mean((y_pred - y) ** 2)
                    candidates.append((expr, mse))
            except:
                pass
        
        # 排序并限制数量
        candidates.sort(key=lambda x: x[1])
        return candidates[:self.config.psrn_top_k]
    
    def _build_initial_population(
        self,
        candidates: List[Tuple[str, float]],
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]],
    ) -> List[Node]:
        """用 PSRN 候选构建 GP 初始种群."""
        from ..utils import string_to_tree
        
        population = []
        
        # 将 PSRN 发现的表达式转换为树
        for expr, mse in candidates[:20]:  # 取前20个
            try:
                tree = string_to_tree(expr, variable_names or ['x'])
                if tree:
                    population.append(tree)
            except:
                pass
        
        # 补充随机个体以增加多样性
        n_diversity = int(len(population) * self.config.diversity_boost)
        for _ in range(n_diversity):
            # 创建随机树
            tree = self._create_random_tree(variable_names or ['x'], max_depth=3)
            population.append(tree)
        
        # 填充到目标种群大小
        while len(population) < self.config.gp_population_size:
            if candidates:
                # 从候选中随机选择并变异
                base_expr = candidates[np.random.randint(len(candidates))][0]
                try:
                    tree = string_to_tree(base_expr, variable_names or ['x'])
                    if tree and np.random.random() < 0.5:
                        tree = self._mutate_tree(tree, variable_names or ['x'])
                    if tree:
                        population.append(tree)
                except:
                    pass
            else:
                # 完全随机
                tree = self._create_random_tree(variable_names or ['x'], max_depth=3)
                population.append(tree)
        
        return population[:self.config.gp_population_size]
    
    def _create_random_tree(self, variable_names: List[str], max_depth: int = 3) -> Node:
        """创建随机表达式树."""
        from ..eml_v2 import Node
        import random
        
        if max_depth <= 0:
            # 叶子节点
            if random.random() < 0.5:
                return Node('const', value=random.uniform(-3, 3))
            else:
                return Node('var', name=random.choice(variable_names))
        
        # 内部节点
        if random.random() < 0.3:
            # 一元算子
            op = random.choice(['sin', 'cos', 'exp', 'log', 'neg'])
            child = self._create_random_tree(variable_names, max_depth - 1)
            return Node('unary', op=op, children=[child])
        else:
            # 二元算子
            op = random.choice(['+', '-', '*', '/', 'eml'])
            left = self._create_random_tree(variable_names, max_depth - 1)
            right = self._create_random_tree(variable_names, max_depth - 1)
            return Node('binary', op=op, children=[left, right])
    
    def _mutate_tree(self, tree: Node, variable_names: List[str]) -> Node:
        """简单变异."""
        import random
        import copy
        
        mutated = copy.deepcopy(tree)
        
        # 随机选择变异点
        nodes = self._get_all_nodes(mutated)
        if not nodes:
            return tree
        
        node = random.choice(nodes)
        
        # 随机变异类型
        if node.node_type == 'const':
            node.value = random.uniform(-3, 3)
        elif node.node_type == 'var':
            node.name = random.choice(variable_names)
        elif node.node_type in ['unary', 'binary']:
            if node.node_type == 'unary':
                node.op = random.choice(['sin', 'cos', 'exp', 'log', 'neg'])
            else:
                node.op = random.choice(['+', '-', '*', '/', 'eml'])
        
        return mutated
    
    def _get_all_nodes(self, tree: Node) -> List[Node]:
        """获取树中所有节点."""
        nodes = [tree]
        if tree.children:
            for child in tree.children:
                nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测."""
        if self.best_tree is None:
            raise RuntimeError("Model not fitted yet")
        
        from ..eml_v2 import evaluate_tree
        return evaluate_tree(self.best_tree, X)
    
    def get_best_expression(self) -> str:
        """获取最优表达式字符串."""
        if self.best_tree is None:
            return ""
        return str(self.best_tree)
