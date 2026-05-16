"""Demo script for PSRN integration with Math Anything.

展示如何在现有 math-anything 框架中使用 PSRN 进行符号回归。
"""

import numpy as np
from math_anything import MathAnything


def demo_basic_psrn():
    """基础演示：使用 PSRN 发现简单方程."""
    print("=" * 60)
    print("Demo 1: Basic PSRN Symbolic Regression")
    print("=" * 60)

    from .bridge import PSRNSymbolicRegression

    # 生成数据: y = x^2 + sin(x)
    x = np.linspace(0, 2, 50)
    y = x**2 + np.sin(x)

    print(f"Target: y = x^2 + sin(x)")
    print(f"Data: {len(x)} samples")

    # 使用 PSRN
    sr = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    best_tree = sr.fit(x.reshape(-1, 1), y, variable_names=["x"])

    print(f"\nDiscovered: {sr._best_expr}")
    print(f"Best MSE: {sr.best_fitness_:.6e}")

    # 查看 Pareto 前沿
    pareto = sr.get_pareto_front()
    print(f"\nPareto Front ({len(pareto)} solutions):")
    for expr, mse, comp, reward in pareto[:5]:
        print(f"  {expr:<30} MSE={mse:.4e}  C={comp}  R={reward:.4f}")


def demo_compare_methods():
    """对比演示：PSRN vs 传统 GP."""
    print("\n" + "=" * 60)
    print("Demo 2: PSRN vs Traditional GP")
    print("=" * 60)

    from ..eml_v2 import ImprovedSymbolicRegression
    from .bridge import PSRNSymbolicRegression

    # 生成数据: y = exp(-x) * cos(2x)
    x = np.linspace(0, 3, 40)
    y = np.exp(-x) * np.cos(2 * x)

    print(f"Target: y = exp(-x) * cos(2x)")

    # 传统 GP
    print("\n--- Traditional GP ---")
    gp = ImprovedSymbolicRegression(population_size=100, generations=30, max_depth=4)
    import time

    t0 = time.time()
    gp_tree = gp.fit(x.reshape(-1, 1), y, variable_names=["x"])
    gp_time = time.time() - t0
    print(f"Result: {gp_tree.to_standard_form()}")
    print(f"Time: {gp_time:.2f}s")
    print(f"MSE: {gp.best_fitness_:.6e}")

    # PSRN
    print("\n--- PSRN ---")
    psrn = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    t0 = time.time()
    psrn_tree = psrn.fit(x.reshape(-1, 1), y, variable_names=["x"])
    psrn_time = time.time() - t0
    print(f"Result: {psrn._best_expr}")
    print(f"Time: {psrn_time:.2f}s")
    print(f"MSE: {psrn.best_fitness_:.6e}")

    print(f"\nSpeedup: {gp_time / max(psrn_time, 1e-6):.2f}x")


def demo_math_anything_integration():
    """演示与 MathAnything API 的集成."""
    print("\n" + "=" * 60)
    print("Demo 3: MathAnything API with PSRN")
    print("=" * 60)

    ma = MathAnything()

    # 从数据发现方程（内部使用 PSRN）
    x = np.linspace(0, 1, 30)
    y = 2 * x**2 - x + 1

    print(f"Target: y = 2x^2 - x + 1")

    # 可以通过 kwargs 传递 PSRN 参数
    equation = ma.discover(
        x.reshape(-1, 1),
        y,
        variable_names=["x"],
        # PSRN 参数（当集成到 api.py 后可用）
    )

    print(f"Discovered: {equation}")


def demo_token_generator():
    """演示 Token Generator 的效果."""
    print("\n" + "=" * 60)
    print("Demo 4: Token Generator")
    print("=" * 60)

    from .token_generator import GPTokenGenerator, MCTSTokenGenerator

    x = np.linspace(0, 1, 20)
    y = np.sin(x) + x**2
    X = x.reshape(-1, 1)

    print("Target: y = sin(x) + x^2")

    # GP Token Generator
    print("\n--- GP Token Generator ---")
    gp_gen = GPTokenGenerator(n_tokens=5)
    gp_tokens, gp_values = gp_gen.generate(X, y, ["x"])
    print(f"Generated {len(gp_tokens)} tokens:")
    for i, token in enumerate(gp_tokens):
        print(f"  {i+1}. {token}")

    # MCTS Token Generator
    print("\n--- MCTS Token Generator ---")
    mcts_gen = MCTSTokenGenerator(n_tokens=5)
    mcts_tokens, mcts_values = mcts_gen.generate(X, y, ["x"])
    print(f"Generated {len(mcts_tokens)} tokens:")
    for i, token in enumerate(mcts_tokens):
        print(f"  {i+1}. {token}")


def demo_pareto_optimization():
    """演示 Pareto 前沿的复杂度-精度权衡."""
    print("\n" + "=" * 60)
    print("Demo 5: Pareto Front Analysis")
    print("=" * 60)

    from .pse_engine import PSEConfig, PSEEngine
    from .psrn_network import PSRNConfig

    # 带噪声的数据
    np.random.seed(42)
    x = np.linspace(0, 2, 100)
    y = x**2 + 0.1 * np.random.randn(100)

    config = PSEConfig(
        psrn_config=PSRNConfig(n_layers=2, n_input_slots=3),
        max_iterations=5,
        reward_discount=0.95,  # 更强的复杂度惩罚
    )
    engine = PSEEngine(config)

    best_expr, pareto = engine.discover(x.reshape(-1, 1), y, ["x"], verbose=False)

    print(f"Best expression: {best_expr}")
    print(f"\nPareto Front (accuracy vs complexity trade-off):")
    print(f"{'Expression':<25} {'MSE':<12} {'Complexity':<10} {'Reward':<10}")
    print("-" * 60)

    for expr, mse, comp, reward in sorted(pareto, key=lambda x: x[2]):
        expr_short = expr[:23] + ".." if len(expr) > 25 else expr
        print(f"{expr_short:<25} {mse:<12.4e} {comp:<10} {reward:<10.4f}")


def demo_memory_estimation():
    """演示内存使用量估算."""
    print("\n" + "=" * 60)
    print("Demo 6: Memory Estimation")
    print("=" * 60)

    from .symbol_layer import SymbolConfig, SymbolLayer

    configs = [
        ("Small", 2, 2),
        ("Medium", 3, 3),
        ("Large", 3, 5),
    ]

    for name, n_layers, n_inputs in configs:
        layer = SymbolLayer()
        dims = layer.memory_estimate(n_inputs, n_layers)

        print(f"\n{name} config ({n_layers} layers, {n_inputs} inputs):")
        for layer_idx, dim in dims.items():
            # 假设 float32，4 bytes
            memory_mb = dim * 4 / (1024 * 1024)
            print(f"  Layer {layer_idx}: {dim:,} expressions ~ {memory_mb:.2f} MB")


if __name__ == "__main__":
    demo_basic_psrn()
    demo_compare_methods()
    demo_math_anything_integration()
    demo_token_generator()
    demo_pareto_optimization()
    demo_memory_estimation()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
