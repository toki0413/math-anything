"""Benchmark - 对比 PSRN 与传统 GP 的性能差异.

测试维度:
1. 搜索空间覆盖能力
2. 计算效率（时间）
3. 发现精度（MSE）
4. 表达式复杂度控制
"""

import time
import numpy as np

from ..eml_v2 import ImprovedSymbolicRegression
from .bridge import PSRNSymbolicRegression


def run_all():
    results = {}

    # ----- Benchmark 1: 简单多项式 -----
    print("\n" + "=" * 60)
    print("Benchmark 1: Simple Polynomial  y = x^2 + 2x + 1")
    print("=" * 60)

    x = np.linspace(-2, 2, 50)
    y = x**2 + 2*x + 1
    X = x.reshape(-1, 1)

    # GP
    t0 = time.time()
    gp = ImprovedSymbolicRegression(population_size=200, generations=50, max_depth=4)
    gp_tree = gp.fit(X, y, variable_names=["x"])
    gp_time = time.time() - t0

    # PSRN
    t0 = time.time()
    psrn = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    psrn_tree = psrn.fit(X, y, variable_names=["x"])
    psrn_time = time.time() - t0

    print(f"GP:  {gp_time:.3f}s  MSE={gp.best_fitness_:.6e}  expr={gp_tree.to_standard_form()}")
    print(f"PSRN: {psrn_time:.3f}s  MSE={psrn.best_fitness_:.6e}  expr={psrn._best_expr}")
    print(f"Speedup: {gp_time/max(psrn_time,1e-6):.1f}x  |  "
          f"MSE ratio: {psrn.best_fitness_/max(gp.best_fitness_,1e-12):.4f}")

    results["polynomial"] = {
        "gp_time": gp_time, "gp_mse": gp.best_fitness_,
        "psrn_time": psrn_time, "psrn_mse": psrn.best_fitness_,
    }

    # ----- Benchmark 2: 三角函数 -----
    print("\n" + "=" * 60)
    print("Benchmark 2: Trigonometric  y = sin(x) + cos(2x)")
    print("=" * 60)

    x = np.linspace(0, 2*np.pi, 60)
    y = np.sin(x) + np.cos(2*x)
    X = x.reshape(-1, 1)

    t0 = time.time()
    gp2 = ImprovedSymbolicRegression(population_size=200, generations=50, max_depth=5)
    gp2.fit(X, y, variable_names=["x"])
    gp2_time = time.time() - t0

    t0 = time.time()
    psrn2 = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    psrn2.fit(X, y, variable_names=["x"])
    psrn2_time = time.time() - t0

    print(f"GP:  {gp2_time:.3f}s  MSE={gp2.best_fitness_:.6e}")
    print(f"PSRN: {psrn2_time:.3f}s  MSE={psrn2.best_fitness_:.6e}")
    print(f"Speedup: {gp2_time/max(psrn2_time,1e-6):.1f}x")

    results["trigonometric"] = {
        "gp_time": gp2_time, "gp_mse": gp2.best_fitness_,
        "psrn_time": psrn2_time, "psrn_mse": psrn2.best_fitness_,
    }

    # ----- Benchmark 3: 指数衰减 -----
    print("\n" + "=" * 60)
    print("Benchmark 3: Exponential Decay  y = exp(-x) * sin(x)")
    print("=" * 60)

    x = np.linspace(0, 5, 50)
    y = np.exp(-x) * np.sin(x)
    X = x.reshape(-1, 1)

    t0 = time.time()
    gp3 = ImprovedSymbolicRegression(population_size=200, generations=50, max_depth=5)
    gp3.fit(X, y, variable_names=["x"])
    gp3_time = time.time() - t0

    t0 = time.time()
    psrn3 = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    psrn3.fit(X, y, variable_names=["x"])
    psrn3_time = time.time() - t0

    print(f"GP:  {gp3_time:.3f}s  MSE={gp3.best_fitness_:.6e}")
    print(f"PSRN: {psrn3_time:.3f}s  MSE={psrn3.best_fitness_:.6e}")
    print(f"Speedup: {gp3_time/max(psrn3_time,1e-6):.1f}x")

    results["exponential"] = {
        "gp_time": gp3_time, "gp_mse": gp3.best_fitness_,
        "psrn_time": psrn3_time, "psrn_mse": psrn3.best_fitness_,
    }

    # ----- Benchmark 4: 搜索空间覆盖 -----
    print("\n" + "=" * 60)
    print("Benchmark 4: Search Space Coverage")
    print("=" * 60)

    from .psrn_network import PSRN, PSRNConfig

    configs = [
        ("Small  (2 inputs, 1 layer)", PSRNConfig(n_layers=1, n_input_slots=2)),
        ("Medium (3 inputs, 2 layers)", PSRNConfig(n_layers=2, n_input_slots=3)),
        ("Large  (4 inputs, 2 layers)", PSRNConfig(n_layers=2, n_input_slots=4)),
    ]

    gp_equivalent = 200 * 50  # GP: pop * generations

    for name, config in configs:
        psrn = PSRN(config)
        sizes = psrn.get_search_space_size()
        total = sum(sizes.values())
        print(f"\n{name}:")
        for layer, size in sizes.items():
            print(f"  Layer {layer}: {size:>10,} expressions")
        print(f"  Total:       {total:>10,}  ({total/gp_equivalent:.1f}x vs GP)")
        print(f"  GP equiv:    {gp_equivalent:>10,}")

    # ----- Benchmark 5: 可扩展性 -----
    print("\n" + "=" * 60)
    print("Benchmark 5: Scalability (sample size)")
    print("=" * 60)
    print(f"{'N samples':>10} | {'GP time':>10} | {'PSRN time':>10} | {'Speedup':>8}")
    print("-" * 48)

    for n in [20, 50, 100, 200]:
        x = np.linspace(-2, 2, n)
        y = x**2 + 2*x + 1
        X = x.reshape(-1, 1)

        t0 = time.time()
        gp_s = ImprovedSymbolicRegression(population_size=200, generations=30, max_depth=4)
        gp_s.fit(X, y, variable_names=["x"])
        gp_s_t = time.time() - t0

        t0 = time.time()
        psrn_s = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
        psrn_s.fit(X, y, variable_names=["x"])
        psrn_s_t = time.time() - t0

        sp = gp_s_t / max(psrn_s_t, 1e-6)
        print(f"{n:>10} | {gp_s_t:>10.3f}s | {psrn_s_t:>10.3f}s | {sp:>7.1f}x")

    # ----- Benchmark 6: 多变量 -----
    print("\n" + "=" * 60)
    print("Benchmark 6: Multi-variable  z = x * y + x^2")
    print("=" * 60)

    np.random.seed(0)
    x1 = np.linspace(-1, 1, 40)
    x2 = np.linspace(-1, 1, 40)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    z = (X1 * X2 + X1**2).ravel()

    t0 = time.time()
    gp_m = ImprovedSymbolicRegression(population_size=200, generations=30, max_depth=4)
    gp_m.fit(X, z, variable_names=["x1", "x2"])
    gp_m_time = time.time() - t0

    t0 = time.time()
    psrn_m = PSRNSymbolicRegression(n_layers=2, max_iterations=3)
    psrn_m.fit(X, z, variable_names=["x1", "x2"])
    psrn_m_time = time.time() - t0

    print(f"GP:  {gp_m_time:.3f}s  MSE={gp_m.best_fitness_:.6e}")
    print(f"PSRN: {psrn_m_time:.3f}s  MSE={psrn_m.best_fitness_:.6e}")
    print(f"Speedup: {gp_m_time/max(psrn_m_time,1e-6):.1f}x")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':>18} | {'GP time':>9} | {'PSRN time':>9} | {'Speedup':>8} | {'GP MSE':>12} | {'PSRN MSE':>12}")
    print("-" * 80)
    for label, r in results.items():
        sp = r["gp_time"] / max(r["psrn_time"], 1e-6)
        print(f"{label:>18} | {r['gp_time']:>8.3f}s | {r['psrn_time']:>8.3f}s | {sp:>7.1f}x | {r['gp_mse']:>11.2e} | {r['psrn_mse']:>11.2e}")

    return results


if __name__ == "__main__":
    run_all()
