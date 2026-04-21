"""Multi-variable relationship discovery for Math Anything.

Optimizes symbolic regression for discovering relationships between
multiple variables (x*y, x/y, x+y, etc.)

Example:
    >>> from math_anything.multivar import MultiVariableDiscovery
    >>> mvd = MultiVariableDiscovery()
    >>>
    >>> # Discover relationship z = x * y + sin(x)
    >>> x = np.linspace(1, 5, 50)
    >>> y = np.linspace(1, 5, 50)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = X * Y + np.sin(X)
    >>>
    >>> result = mvd.discover(X.flatten(), Y.flatten(), Z.flatten())
    >>> print(result)
    "(x * y) + sin(x)"
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .eml_v2 import ExprBuilder, ImprovedSymbolicRegression, Node, NodeType, safe_div


class MultiVariableDiscovery:
    """Discover multi-variable mathematical relationships.

    This class extends symbolic regression with:
    - Pre-defined interaction templates
    - Correlation-based variable selection
    - Multi-variable expression initialization
    """

    def __init__(
        self,
        population_size: int = 150,
        max_depth: int = 6,
        generations: int = 100,
        use_templates: bool = True,
        correlation_threshold: float = 0.1,
    ):
        self.population_size = population_size
        self.max_depth = max_depth
        self.generations = generations
        self.use_templates = use_templates
        self.correlation_threshold = correlation_threshold
        self.builder = ExprBuilder()

    def discover(
        self,
        *arrays: np.ndarray,
        target: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
    ) -> str:
        """Discover relationship between multiple variables.

        Args:
            *arrays: Input variable arrays (can be 1D or flattened)
            target: Target values (if None, last array is target)
            variable_names: Names of variables

        Returns:
            Discovered equation as string

        Example:
            >>> mvd = MultiVariableDiscovery()
            >>>
            >>> # Discover z = x^2 + y^2
            >>> x = np.linspace(-2, 2, 30)
            >>> y = np.linspace(-2, 2, 30)
            >>> X, Y = np.meshgrid(x, y)
            >>> Z = X**2 + Y**2
            >>>
            >>> eq = mvd.discover(X, Y, target=Z, variable_names=['x', 'y'])
            >>> print(eq)
            "(x^2 + y^2)"
        """
        # Process input arrays
        arrays = [np.array(a).flatten() for a in arrays]

        if target is None:
            # Last array is target
            if len(arrays) < 2:
                raise ValueError("Need at least 2 arrays if target is not specified")
            target = arrays[-1]
            arrays = arrays[:-1]
        else:
            target = np.array(target).flatten()

        n_vars = len(arrays)

        # Set variable names
        if variable_names is None:
            self.variable_names = [f"x{i}" for i in range(n_vars)]
        else:
            self.variable_names = variable_names[:n_vars]

        # Build feature matrix
        X = np.column_stack(arrays)

        # Analyze correlations
        correlations = self._analyze_correlations(X, target)
        print(f"Variable correlations with target: {correlations}")

        # Create templates based on correlations
        templates = []
        if self.use_templates:
            templates = self._create_templates(correlations)
            print(f"Created {len(templates)} templates")

        # Run symbolic regression with templates
        sr = ImprovedSymbolicRegression(
            population_size=self.population_size,
            max_depth=self.max_depth,
            generations=self.generations,
        )

        # Initialize population with templates
        population = self._init_with_templates(sr, templates, n_vars)

        # Run evolution
        best_tree = self._evolve_with_templates(sr, population, X, target)

        return best_tree.to_standard_form()

    def _analyze_correlations(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Analyze correlation between each variable and target."""
        correlations = {}
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations[i] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[i] = 0.0
        return correlations

    def _create_templates(self, correlations: Dict[int, float]) -> List[Node]:
        """Create expression templates based on correlations."""
        templates = []

        # Get most important variables
        important_vars = [
            i for i, corr in correlations.items() if corr > self.correlation_threshold
        ]

        if len(important_vars) == 0:
            important_vars = list(correlations.keys())

        # Single variable templates
        for i in important_vars:
            var = self.builder.var(self.variable_names[i])

            # Linear: x
            templates.append(var)

            # Quadratic: x^2
            templates.append(self.builder.pow(var, self.builder.const(2)))

            # Square root: sqrt(x)
            templates.append(self.builder.sqrt(var))

            # Sine: sin(x)
            templates.append(self.builder.sin(var))

            # Cosine: cos(x)
            templates.append(self.builder.cos(var))

        # Two-variable interaction templates
        if len(important_vars) >= 2:
            for i in important_vars:
                for j in important_vars:
                    if i >= j:  # Avoid duplicates
                        continue

                    var_i = self.builder.var(self.variable_names[i])
                    var_j = self.builder.var(self.variable_names[j])

                    # Product: x * y
                    templates.append(self.builder.mul(var_i, var_j))

                    # Division: x / y
                    templates.append(self.builder.div(var_i, var_j))

                    # Sum: x + y
                    templates.append(self.builder.add(var_i, var_j))

                    # Difference: x - y
                    templates.append(self.builder.sub(var_i, var_j))

        # Three-variable templates (if enough variables)
        if len(important_vars) >= 3:
            for i in important_vars[:3]:
                for j in important_vars[:3]:
                    for k in important_vars[:3]:
                        if i >= j or j >= k:
                            continue

                        var_i = self.builder.var(self.variable_names[i])
                        var_j = self.builder.var(self.variable_names[j])
                        var_k = self.builder.var(self.variable_names[k])

                        # x * y * z
                        prod_ij = self.builder.mul(var_i, var_j)
                        templates.append(self.builder.mul(prod_ij, var_k))

        return templates

    def _init_with_templates(
        self, sr: ImprovedSymbolicRegression, templates: List[Node], n_vars: int
    ) -> List[Node]:
        """Initialize population with templates."""
        sr.variables = self.variable_names
        sr.builder = self.builder

        population = []

        # Add templates first
        for template in templates[: self.population_size // 3]:
            population.append(template.copy())

        # Fill rest with random trees
        while len(population) < self.population_size:
            tree = sr._random_tree_grow(sr.max_depth)
            population.append(tree)

        return population

    def _evolve_with_templates(
        self,
        sr: ImprovedSymbolicRegression,
        population: List[Node],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Node:
        """Run evolution with template-initialized population."""
        best_tree = None
        best_fitness = float("inf")
        no_improvement_count = 0

        for gen in range(sr.generations):
            # Evaluate fitness
            fitness = [sr._fitness(tree, X, y) for tree in population]

            # Track best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_tree = population[gen_best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if best_fitness < 1e-12:
                print(f"Gen {gen}: Perfect fit!")
                break

            if no_improvement_count > 25:
                print(f"Gen {gen}: No improvement, stopping")
                break

            # Evolve
            population = sr._evolve(population, fitness)

            if gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {best_fitness:.6f}")

        if best_tree is None:
            return self.builder.var(self.variable_names[0])

        return best_tree


def discover_multivar(
    *arrays: np.ndarray,
    target: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Convenience function for multi-variable discovery.

    Example:
        >>> # Discover z = x * y
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 3, 4, 5, 6])
        >>> z = x * y  # [2, 6, 12, 20, 30]
        >>>
        >>> eq = discover_multivar(x, y, target=z, variable_names=['x', 'y'])
        >>> print(eq)  # "(x * y)"
    """
    mvd = MultiVariableDiscovery(**kwargs)
    return mvd.discover(*arrays, target=target, variable_names=variable_names)


def analyze_interactions(
    X: np.ndarray, y: np.ndarray, variable_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze variable interactions without full discovery.

    Returns statistics about variable correlations and suggested templates.

    Example:
        >>> interactions = analyze_interactions(X, y, ['x', 'y', 'z'])
        >>> print(interactions['correlations'])
        >>> print(interactions['suggested_templates'])
    """
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(X.shape[1])]

    # Calculate correlations
    correlations = {}
    for i in range(X.shape[1]):
        if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations[variable_names[i]] = corr

    # Calculate pairwise correlations
    pairwise = {}
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            # Product correlation
            prod = X[:, i] * X[:, j]
            if np.std(prod) > 1e-10:
                corr = np.corrcoef(prod, y)[0, 1]
                pairwise[f"{variable_names[i]}*{variable_names[j]}"] = corr

            # Ratio correlation
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(X[:, j]) > 1e-10, X[:, i] / X[:, j], np.nan)
            if np.std(ratio) > 1e-10 and not np.all(np.isnan(ratio)):
                corr = np.corrcoef(ratio, y)[0, 1]
                pairwise[f"{variable_names[i]}/{variable_names[j]}"] = corr

    # Find strongest interactions
    all_corrs = {**{f"{k}": v for k, v in correlations.items()}, **pairwise}
    sorted_corrs = sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "correlations": correlations,
        "pairwise_interactions": pairwise,
        "strongest_predictors": sorted_corrs[:5],
        "suggested_templates": [t for t, c in sorted_corrs[:3]],
    }


__all__ = [
    "MultiVariableDiscovery",
    "discover_multivar",
    "analyze_interactions",
]
