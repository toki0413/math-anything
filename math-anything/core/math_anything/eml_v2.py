"""EML (Exp-Minus-Log) Operator - Improved Version with Numerical Stability.

Based on: "All elementary functions from a single binary operator"
by Andrzej Odrzywołek, arXiv:2603.21852

Improvements:
- Numerical stability with safe evaluation
- Better genetic operators (crossover, mutation)
- Protected operators to handle edge cases
- More robust fitness evaluation
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Constants for numerical stability
MAX_VALUE = 1e10
MIN_VALUE = -1e10
EPSILON = 1e-10


def safe_exp(x: float) -> float:
    """Safe exponential with clipping."""
    x = np.clip(x, -700, 700)  # Prevent overflow
    return np.exp(x)


def safe_log(x: float) -> float:
    """Safe logarithm with protection."""
    if x <= 0:
        return -100.0  # Large negative penalty
    return np.log(max(x, EPSILON))


def safe_div(a: float, b: float) -> float:
    """Safe division."""
    if abs(b) < EPSILON:
        return MAX_VALUE if a >= 0 else MIN_VALUE
    return a / b


class NodeType(Enum):
    """Types of nodes in expression tree."""

    EML = auto()  # eml(x, y) = exp(x) - ln(y)
    ADD = auto()  # x + y
    SUB = auto()  # x - y
    MUL = auto()  # x * y
    DIV = auto()  # x / y
    SIN = auto()  # sin(x)
    COS = auto()  # cos(x)
    POW = auto()  # x^y
    SQRT = auto()  # sqrt(x)
    ABS = auto()  # abs(x)
    VAR = auto()  # Variable
    CONST = auto()  # Constant


@dataclass
class Node:
    """A node in the expression tree."""

    node_type: NodeType
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None  # For CONST
    name: Optional[str] = None  # For VAR

    def evaluate(self, variables: Dict[str, float]) -> float:
        """Evaluate expression tree with numerical protection."""
        try:
            if self.node_type == NodeType.CONST:
                return self.value if self.value is not None else 0.0

            elif self.node_type == NodeType.VAR:
                if self.name not in variables:
                    raise ValueError(f"Variable '{self.name}' not provided")
                return variables[self.name]

            elif self.node_type == NodeType.EML:
                left_val = self.left.evaluate(variables)
                right_val = self.right.evaluate(variables)
                # eml(x, y) = exp(x) - ln(y)
                return safe_exp(left_val) - safe_log(right_val)

            elif self.node_type == NodeType.ADD:
                return self.left.evaluate(variables) + self.right.evaluate(variables)

            elif self.node_type == NodeType.SUB:
                return self.left.evaluate(variables) - self.right.evaluate(variables)

            elif self.node_type == NodeType.MUL:
                return self.left.evaluate(variables) * self.right.evaluate(variables)

            elif self.node_type == NodeType.DIV:
                a = self.left.evaluate(variables)
                b = self.right.evaluate(variables)
                return safe_div(a, b)

            elif self.node_type == NodeType.SIN:
                return np.sin(self.left.evaluate(variables))

            elif self.node_type == NodeType.COS:
                return np.cos(self.left.evaluate(variables))

            elif self.node_type == NodeType.POW:
                base = self.left.evaluate(variables)
                exp = self.right.evaluate(variables)
                # Protect against invalid powers
                if base < 0 and not float(exp).is_integer():
                    return float("nan")
                # Prevent overflow/underflow before computation
                if abs(base) > 100 and abs(exp) > 5:
                    return float("nan")
                if abs(base) < EPSILON and exp < 0:
                    return float("nan")
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        result = np.power(base, exp)
                    if np.isinf(result) or np.isnan(result):
                        return float("nan")
                    return float(result)
                except (OverflowError, ValueError, FloatingPointError):
                    return float("nan")

            elif self.node_type == NodeType.SQRT:
                val = self.left.evaluate(variables)
                if val < 0:
                    return float("nan")
                return np.sqrt(val)

            elif self.node_type == NodeType.ABS:
                return abs(self.left.evaluate(variables))

            else:
                return 0.0
        except Exception:
            return float("nan")

    def to_string(self) -> str:
        """Convert to string representation."""
        if self.node_type == NodeType.CONST:
            return f"{self.value:.4g}" if self.value is not None else "0"

        elif self.node_type == NodeType.VAR:
            return self.name or "x"

        elif self.node_type == NodeType.EML:
            return f"eml({self.left.to_string()}, {self.right.to_string()})"

        elif self.node_type == NodeType.ADD:
            return f"({self.left.to_string()} + {self.right.to_string()})"

        elif self.node_type == NodeType.SUB:
            return f"({self.left.to_string()} - {self.right.to_string()})"

        elif self.node_type == NodeType.MUL:
            return f"({self.left.to_string()} * {self.right.to_string()})"

        elif self.node_type == NodeType.DIV:
            return f"({self.left.to_string()} / {self.right.to_string()})"

        elif self.node_type == NodeType.SIN:
            return f"sin({self.left.to_string()})"

        elif self.node_type == NodeType.COS:
            return f"cos({self.left.to_string()})"

        elif self.node_type == NodeType.POW:
            return f"({self.left.to_string()}^{self.right.to_string()})"

        elif self.node_type == NodeType.SQRT:
            return f"sqrt({self.left.to_string()})"

        elif self.node_type == NodeType.ABS:
            return f"abs({self.left.to_string()})"

        else:
            return "?"

    def to_standard_form(self, simplify_tree: bool = True) -> str:
        """Convert to standard mathematical notation.

        Args:
            simplify_tree: Whether to apply algebraic simplification

        Returns:
            String representation of the equation
        """
        if simplify_tree:
            from .simplifier import simplify

            simplified = simplify(self)
            return simplified._simplify()
        return self._simplify()

    def _simplify(self) -> str:
        """Simplify to readable form."""
        if self.node_type == NodeType.CONST:
            if self.value is None:
                return "0"
            if abs(self.value - 1.0) < 1e-10:
                return "1"
            if abs(self.value - math.e) < 1e-10:
                return "e"
            if abs(self.value) < 1e-10:
                return "0"
            return f"{self.value:.4g}"

        elif self.node_type == NodeType.VAR:
            return self.name or "x"

        elif self.node_type == NodeType.EML:
            left = self.left
            right = self.right

            # exp(x) pattern: eml(x, 1)
            if (
                right.node_type == NodeType.CONST
                and right.value is not None
                and abs(right.value - 1.0) < 1e-10
            ):
                return f"exp({left._simplify()})"

            # ln(x) pattern detection is complex, skip for now
            return f"eml({left._simplify()}, {right._simplify()})"

        elif self.node_type == NodeType.ADD:
            return f"({self.left._simplify()} + {self.right._simplify()})"

        elif self.node_type == NodeType.SUB:
            return f"({self.left._simplify()} - {self.right._simplify()})"

        elif self.node_type == NodeType.MUL:
            return f"({self.left._simplify()} * {self.right._simplify()})"

        elif self.node_type == NodeType.DIV:
            return f"({self.left._simplify()} / {self.right._simplify()})"

        elif self.node_type == NodeType.SIN:
            return f"sin({self.left._simplify()})"

        elif self.node_type == NodeType.COS:
            return f"cos({self.left._simplify()})"

        elif self.node_type == NodeType.POW:
            return f"({self.left._simplify()}^{self.right._simplify()})"

        elif self.node_type == NodeType.SQRT:
            return f"sqrt({self.left._simplify()})"

        elif self.node_type == NodeType.ABS:
            return f"abs({self.left._simplify()})"

        return "?"

    def depth(self) -> int:
        """Get tree depth."""
        if self.node_type in (NodeType.CONST, NodeType.VAR):
            return 0
        left_d = self.left.depth() if self.left else 0
        right_d = self.right.depth() if self.right else 0
        return 1 + max(left_d, right_d)

    def node_count(self) -> int:
        """Count nodes in tree."""
        if self.node_type in (NodeType.CONST, NodeType.VAR):
            return 1
        count = 1
        if self.left:
            count += self.left.node_count()
        if self.right:
            count += self.right.node_count()
        return count

    def copy(self) -> "Node":
        """Deep copy."""
        if self.node_type == NodeType.CONST:
            return Node(NodeType.CONST, value=self.value)
        elif self.node_type == NodeType.VAR:
            return Node(NodeType.VAR, name=self.name)
        else:
            return Node(
                self.node_type,
                left=self.left.copy() if self.left else None,
                right=self.right.copy() if self.right else None,
            )

    def get_all_nodes(self) -> List[Tuple["Node", Optional["Node"], str]]:
        """Get all nodes with their parents and direction.

        Returns list of (node, parent, direction) where direction is 'left' or 'right'
        """
        nodes = [(self, None, "")]
        if self.left:
            nodes.extend([(n, p, d) for n, p, d in self.left.get_all_nodes()])
            nodes[-len(self.left.get_all_nodes()) :][0] = (self.left, self, "left")
        if self.right:
            nodes.extend([(n, p, d) for n, p, d in self.right.get_all_nodes()])
            nodes[-len(self.right.get_all_nodes()) :][0] = (self.right, self, "right")
        return nodes


class ExprBuilder:
    """Builder for expression trees."""

    @staticmethod
    def const(value: float) -> Node:
        return Node(NodeType.CONST, value=value)

    @staticmethod
    def var(name: str) -> Node:
        return Node(NodeType.VAR, name=name)

    @staticmethod
    def eml(x: Node, y: Node) -> Node:
        return Node(NodeType.EML, left=x, right=y)

    @staticmethod
    def add(x: Node, y: Node) -> Node:
        return Node(NodeType.ADD, left=x, right=y)

    @staticmethod
    def sub(x: Node, y: Node) -> Node:
        return Node(NodeType.SUB, left=x, right=y)

    @staticmethod
    def mul(x: Node, y: Node) -> Node:
        return Node(NodeType.MUL, left=x, right=y)

    @staticmethod
    def div(x: Node, y: Node) -> Node:
        return Node(NodeType.DIV, left=x, right=y)

    @classmethod
    def exp(cls, x: Node) -> Node:
        """exp(x) = eml(x, 1)"""
        return cls.eml(x, cls.const(1.0))

    @classmethod
    def ln(cls, x: Node) -> Node:
        """Approximate ln using eml"""
        # ln(x) ≈ eml(0, x) - 1 for x near 1
        return cls.sub(cls.eml(cls.const(0.0), x), cls.const(1.0))

    @staticmethod
    def sin(x: Node) -> Node:
        """sin(x)"""
        return Node(NodeType.SIN, left=x)

    @staticmethod
    def cos(x: Node) -> Node:
        """cos(x)"""
        return Node(NodeType.COS, left=x)

    @staticmethod
    def pow(x: Node, y: Node) -> Node:
        """x^y"""
        return Node(NodeType.POW, left=x, right=y)

    @staticmethod
    def sqrt(x: Node) -> Node:
        """sqrt(x)"""
        return Node(NodeType.SQRT, left=x)

    @staticmethod
    def abs(x: Node) -> Node:
        """abs(x)"""
        return Node(NodeType.ABS, left=x)


class ImprovedSymbolicRegression:
    """Improved symbolic regression with better genetic operators."""

    def __init__(
        self,
        population_size: int = 200,
        max_depth: int = 5,
        generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1,
        use_standard_ops: bool = True,
    ):
        self.population_size = population_size
        self.max_depth = max_depth
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.use_standard_ops = use_standard_ops  # Use +, -, *, / in addition to EML
        self.variables: List[str] = []
        self.builder = ExprBuilder()
        self.best_tree_: Optional[Node] = None
        self.best_fitness_: float = float("inf")

    def fit(
        self, X: np.ndarray, y: np.ndarray, variable_names: Optional[List[str]] = None
    ) -> Node:
        """Discover equation from data."""
        n_features = X.shape[1]
        if variable_names is None:
            self.variables = [f"x{i}" for i in range(n_features)]
        else:
            self.variables = variable_names[:n_features]

        # Initialize population
        population = self._init_population()

        best_tree = None
        best_fitness = float("inf")
        no_improvement_count = 0

        for gen in range(self.generations):
            # Evaluate fitness
            fitness = [self._fitness(tree, X, y) for tree in population]

            # Track best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_tree = population[gen_best_idx].copy()
                self.best_fitness_ = best_fitness
                self.best_tree_ = best_tree.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if best_fitness < 1e-12:
                print(f"Gen {gen}: Perfect fit found!")
                break

            if no_improvement_count > 20:
                print(f"Gen {gen}: No improvement for 20 generations, stopping")
                break

            # Evolve
            population = self._evolve(population, fitness)

            if gen % 10 == 0 or gen < 5:
                print(
                    f"Gen {gen}: Best fitness = {best_fitness:.6f}, "
                    f"Nodes = {best_tree.node_count() if best_tree else 0}"
                )

        if best_tree is None:
            return (
                self.builder.var(self.variables[0])
                if self.variables
                else self.builder.const(0.0)
            )

        return best_tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best discovered tree.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted values
        """
        if self.best_tree_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._evaluate_tree(self.best_tree_, X)

    def _evaluate_tree(self, tree: Node, X: np.ndarray) -> np.ndarray:
        """Evaluate tree on input data."""
        if tree.node_type == NodeType.CONST:
            return np.full(X.shape[0], tree.value)

        if tree.node_type == NodeType.VAR:
            var_idx = (
                self.variables.index(tree.name) if tree.name in self.variables else 0
            )
            return X[:, var_idx]

        if tree.node_type == NodeType.ADD:
            return self._evaluate_tree(tree.left, X) + self._evaluate_tree(
                tree.right, X
            )

        if tree.node_type == NodeType.SUB:
            return self._evaluate_tree(tree.left, X) - self._evaluate_tree(
                tree.right, X
            )

        if tree.node_type == NodeType.MUL:
            return self._evaluate_tree(tree.left, X) * self._evaluate_tree(
                tree.right, X
            )

        if tree.node_type == NodeType.DIV:
            left = self._evaluate_tree(tree.left, X)
            right = self._evaluate_tree(tree.right, X)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(right) > 1e-10, left / right, 0.0)

        if tree.node_type == NodeType.SIN:
            return np.sin(self._evaluate_tree(tree.left, X))

        if tree.node_type == NodeType.COS:
            return np.cos(self._evaluate_tree(tree.left, X))

        if tree.node_type == NodeType.SQRT:
            val = self._evaluate_tree(tree.left, X)
            return np.sqrt(np.abs(val))

        if tree.node_type == NodeType.ABS:
            return np.abs(self._evaluate_tree(tree.left, X))

        if tree.node_type == NodeType.EML:
            left = self._evaluate_tree(tree.left, X)
            right = self._evaluate_tree(tree.right, X)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.exp(left - np.log(np.abs(right) + 1e-10))
            return np.where(np.isfinite(result), result, 0.0)

        if tree.node_type == NodeType.POW:
            base = self._evaluate_tree(tree.left, X)
            exp = self._evaluate_tree(tree.right, X)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result = np.power(np.abs(base), exp)
            return np.where(np.isfinite(result), result, 0.0)

        return np.zeros(X.shape[0])

    def _init_population(self) -> List[Node]:
        """Initialize with random trees using ramped half-and-half."""
        population = []

        # Half full trees, half grow trees
        for i in range(self.population_size):
            if i < self.population_size // 2:
                tree = self._random_tree_full(
                    min(i % self.max_depth + 1, self.max_depth)
                )
            else:
                tree = self._random_tree_grow(
                    min(i % self.max_depth + 1, self.max_depth)
                )
            population.append(tree)

        return population

    def _random_tree_full(self, depth: int) -> Node:
        """Generate full tree of given depth."""
        if depth == 0:
            return self._random_terminal()

        op = self._random_operator()

        # Handle unary operators
        unary_ops = {NodeType.SIN, NodeType.COS, NodeType.SQRT, NodeType.ABS}
        if op in unary_ops:
            left = self._random_tree_full(depth - 1)
            return Node(op, left=left)

        # Binary operators
        left = self._random_tree_full(depth - 1)
        right = self._random_tree_full(depth - 1)
        return Node(op, left=left, right=right)

    def _random_tree_grow(self, max_depth: int, current_depth: int = 0) -> Node:
        """Generate tree with grow method."""
        if current_depth >= max_depth:
            return self._random_terminal()

        if current_depth > 0 and random.random() < 0.3:
            return self._random_terminal()

        op = self._random_operator()

        # Handle unary operators
        unary_ops = {NodeType.SIN, NodeType.COS, NodeType.SQRT, NodeType.ABS}
        if op in unary_ops:
            left = self._random_tree_grow(max_depth, current_depth + 1)
            return Node(op, left=left)

        # Binary operators
        left = self._random_tree_grow(max_depth, current_depth + 1)
        right = self._random_tree_grow(max_depth, current_depth + 1)
        return Node(op, left=left, right=right)

    def _random_terminal(self) -> Node:
        """Random terminal node."""
        if random.random() < 0.5:
            return self.builder.const(random.uniform(-5, 5))
        else:
            return self.builder.var(random.choice(self.variables))

    def _random_operator(self) -> NodeType:
        """Random operator."""
        if self.use_standard_ops:
            # Binary operators
            binary_ops = [
                NodeType.EML,
                NodeType.ADD,
                NodeType.SUB,
                NodeType.MUL,
                NodeType.DIV,
                NodeType.POW,
            ]
            binary_weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.1]

            # Unary operators (need special handling in tree generation)
            unary_ops = [NodeType.SIN, NodeType.COS, NodeType.SQRT, NodeType.ABS]
            unary_weights = [0.05, 0.05, 0.03, 0.02]

            all_ops = binary_ops + unary_ops
            all_weights = binary_weights + unary_weights
            return random.choices(all_ops, weights=all_weights)[0]
        else:
            return NodeType.EML

    def _fitness(self, tree: Node, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate fitness with multiple criteria."""
        # Handle NaN/Inf in tree structure
        node_count = tree.node_count()
        if node_count > 50:  # Penalize overly complex trees
            return 1e10

        try:
            predictions = []
            valid_count = 0

            for i in range(len(X)):
                vars_dict = {
                    name: float(X[i, j]) for j, name in enumerate(self.variables)
                }
                pred = tree.evaluate(vars_dict)

                if np.isnan(pred) or np.isinf(pred) or abs(pred) > 1e15:
                    predictions.append(0.0)
                else:
                    predictions.append(pred)
                    valid_count += 1

            predictions = np.array(predictions)

            # If too many invalid predictions, high penalty
            if valid_count < len(X) * 0.5:
                return 1e8 + (len(X) - valid_count)

            # Mean squared error
            mse = np.mean((predictions - y) ** 2)

            # Complexity penalty (Ockham's razor)
            complexity = 0.001 * node_count

            # Correlation bonus (encourage structure matching)
            try:
                if np.std(predictions) > 1e-10 and np.std(y) > 1e-10:
                    corr = np.corrcoef(predictions, y)[0, 1]
                    if not np.isnan(corr):
                        corr_penalty = -0.1 * abs(corr)  # Reward correlation
                    else:
                        corr_penalty = 0
                else:
                    corr_penalty = 0
            except:
                corr_penalty = 0

            fitness = mse + complexity + corr_penalty

            # Ensure non-negative
            return max(fitness, 0)

        except Exception as e:
            return 1e10

    def _evolve(self, population: List[Node], fitness: List[float]) -> List[Node]:
        """Evolve population."""
        new_population = []

        # Elitism
        sorted_indices = np.argsort(fitness)
        elite_count = max(1, int(self.population_size * self.elitism_ratio))
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())

        # Generate rest through selection and reproduction
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population, fitness)

            if random.random() < self.crossover_rate:
                parent2 = self._tournament_select(population, fitness)
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population[: self.population_size]

    def _tournament_select(
        self, population: List[Node], fitness: List[float], tournament_size: int = 5
    ) -> Node:
        """Tournament selection."""
        indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )
        best_idx = min(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _crossover(self, p1: Node, p2: Node) -> Node:
        """Subtree crossover."""
        c1 = p1.copy()

        # Get all nodes from both trees
        nodes1 = c1.get_all_nodes()
        nodes2 = p2.get_all_nodes()

        if len(nodes1) <= 1 or len(nodes2) <= 1:
            return c1

        # Select random crossover points (excluding root)
        point1 = random.choice(nodes1[1:])
        point2 = random.choice(nodes2[1:])

        node1, parent1, dir1 = point1
        node2 = point2[0]

        # Swap subtrees
        if parent1 is not None and dir1:
            if dir1 == "left":
                parent1.left = node2.copy()
            else:
                parent1.right = node2.copy()

        return c1

    def _mutate(self, tree: Node) -> Node:
        """Point mutation or subtree mutation."""
        mutation_type = random.random()

        if mutation_type < 0.3:
            # Point mutation: change operator
            nodes = tree.get_all_nodes()
            if len(nodes) > 1:
                node, parent, direction = random.choice(nodes[1:])
                if node.node_type not in (NodeType.CONST, NodeType.VAR):
                    node.node_type = self._random_operator()

        elif mutation_type < 0.6:
            # Terminal mutation: change constant value
            nodes = tree.get_all_nodes()
            const_nodes = [
                (n, p, d) for n, p, d in nodes if n.node_type == NodeType.CONST
            ]
            if const_nodes:
                node, _, _ = random.choice(const_nodes)
                node.value = random.uniform(-5, 5)

        else:
            # Subtree mutation: replace subtree
            nodes = tree.get_all_nodes()
            if len(nodes) > 1:
                node, parent, direction = random.choice(nodes[1:])
                if parent is not None and direction:
                    new_subtree = self._random_tree_grow(self.max_depth // 2)
                    if direction == "left":
                        parent.left = new_subtree
                    else:
                        parent.right = new_subtree

        return tree


# Backward compatibility
def discover_equation(
    X: np.ndarray, y: np.ndarray, variable_names: Optional[List[str]] = None, **kwargs
) -> str:
    """Discover equation from data."""
    sr = ImprovedSymbolicRegression(**kwargs)
    best_tree = sr.fit(X, y, variable_names)
    return best_tree.to_standard_form()


def eml(x: float, y: float) -> float:
    """EML operator with safety."""
    return safe_exp(x) - safe_log(y)


# Backward compatibility aliases
EMLNode = Node
NodeType = NodeType
SymbolicRegression = ImprovedSymbolicRegression


__all__ = [
    "NodeType",
    "Node",
    "EMLNode",
    "ExprBuilder",
    "ImprovedSymbolicRegression",
    "SymbolicRegression",
    "discover_equation",
    "eml",
    "safe_exp",
    "safe_log",
]
