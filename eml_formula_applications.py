#!/usr/bin/env python3
"""
EML核心公式的实际应用场景
展示如何用三重嵌套EML公式解决实际问题
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, fsolve

print("=" * 78)
print("EML核心公式的实际应用场景")
print("=" * 78)

# 已拟合的参数
a1, b1 = 0.180, 3.240  # 水化增强参数
a2, b2 = 0.010, 1.000  # 孔隙削弱参数
gamma = 0.001  # 温度敏感系数
k = 0.1  # 水化速率常数

def eml(x, y):
    """EML核心算子"""
    return np.exp(x) - np.log(y + 1e-10)

def phi_eml(psi, a1=a1, b1=b1, a2=a2, b2=b2):
    """三重嵌套EML性能预测模型"""
    inner1 = eml(a1 * psi, b1)
    inner2 = eml(a2 * psi, b2)
    return eml(inner1, inner2)

def psi_calc(H, T, gamma=gamma):
    """计算归一化养护状态"""
    return H * np.exp(-gamma * T)

def H_calc(t, T, k=k, Ea=5000, R=8.314):
    """计算水化程度 (Arrhenius修正)"""
    # k_T = k * exp(-Ea/(R*(T+273)))
    k_T = k * np.exp(-Ea/(R*(T+273))) * np.exp(Ea/(R*293))  # 归一化到20℃
    return 1 - np.exp(-k_T * t)

print("\n【应用1】性能预测 - 给定养护条件预测UHPC强度")
print("=" * 78)

print("\n场景：设计UHPC配合比，需要预测不同养护条件下的强度")
print("-" * 78)

test_conditions = [
    (20, 3), (20, 7), (20, 28),
    (40, 3), (40, 7), (40, 28),
    (60, 3), (60, 7), (60, 28),
    (80, 7), (100, 7)
]

print(f"\n{'温度(℃)':<10} {'龄期(d)':<10} {'水化程度H':<12} {'Ψ':<12} {'预测强度比Φ':<15} {'估算强度(MPa)':<15}")
print("-" * 78)

sigma_max = 109.1  # 实验最大强度

for T, t in test_conditions:
    H = H_calc(t, T)
    psi = psi_calc(H, T)
    phi = phi_eml(psi)
    sigma = phi * sigma_max
    print(f"{T:<10} {t:<10} {H:<12.3f} {psi:<12.3f} {phi:<15.3f} {sigma:<15.1f}")

print("\n【应用2】最优养护策略 - 求最大强度的养护条件")
print("=" * 78)

print("\n场景：工程现场需要知道最佳养护温度和时间")
print("-" * 78)

def negative_phi(T, t):
    """用于优化的负性能函数"""
    H = H_calc(t, T)
    psi = psi_calc(H, T)
    return -phi_eml(psi)

# 固定龄期，找最优温度
for age in [3, 7, 14, 28]:
    result = minimize_scalar(
        lambda T: negative_phi(T, age),
        bounds=(10, 100),
        method='bounded'
    )
    T_opt = result.x
    phi_max = -result.fun
    print(f"龄期 {age}d: 最优温度 = {T_opt:.1f}℃, 最大强度比 = {phi_max:.3f}")

# 固定温度，找最优龄期
print("\n固定温度下的最优龄期（理论分析）：")
for T in [20, 40, 60, 80]:
    # 长期强度极限
    H_inf = 1.0
    psi_inf = psi_calc(H_inf, T)
    phi_inf = phi_eml(psi_inf)
    
    # 达到90%强度所需时间
    target_phi = 0.9 * phi_inf
    
    # 求解 t
    def equation(t):
        H = H_calc(t, T)
        psi = psi_calc(H, T)
        return phi_eml(psi) - target_phi
    
    try:
        t_90 = fsolve(equation, 10)[0]
        print(f"温度 {T}℃: 90%强度需 {t_90:.1f}d, 极限强度比 = {phi_inf:.3f}")
    except:
        print(f"温度 {T}℃: 计算失败")

print("\n【应用3】配方优化 - 调整参数适配不同原材料")
print("=" * 78)

print("\n场景：更换水泥品种后，需要重新标定EML参数")
print("-" * 78)

# 模拟新水泥的实验数据
# 假设新水泥水化更快但早期孔隙更多
new_data = [
    {'T': 20, 't': 3, 'phi': 0.75},
    {'T': 20, 't': 7, 'phi': 0.88},
    {'T': 20, 't': 28, 'phi': 0.95},
    {'T': 40, 't': 3, 'phi': 0.82},
    {'T': 40, 't': 7, 'phi': 0.91},
    {'T': 60, 't': 7, 'phi': 0.88},
]

def fit_eml_params(data):
    """拟合EML参数"""
    from scipy.optimize import minimize
    
    def loss(params):
        a1, b1, a2, b2, gamma, k = params
        total_error = 0
        for d in data:
            H = 1 - np.exp(-k * d['t'])
            psi = H * np.exp(-gamma * d['T'])
            phi_pred = phi_eml(psi, a1, b1, a2, b2)
            total_error += (phi_pred - d['phi'])**2
        return total_error
    
    # 初始参数和约束
    x0 = [0.18, 3.2, 0.01, 1.0, 0.001, 0.1]
    bounds = [(0.05, 0.5), (1.5, 5), (0.001, 0.05), (0.5, 2), (0.0001, 0.01), (0.05, 0.3)]
    
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    return result.x

print("\n新水泥标定结果（示例）：")
print("原始参数: a1=0.18, b1=3.24, a2=0.01, b2=1.0, gamma=0.001, k=0.1")
print("新水泥参数: a1=0.22（水化增强更快）, b1=2.8（早期基础更好）...")

print("\n【应用4】质量控制 - 根据实测强度反推实际养护效果")
print("=" * 78)

print("\n场景：现场抽检强度不达标，分析原因")
print("-" * 78)

# 实测数据
measured_cases = [
    {'T_nominal': 60, 't_nominal': 7, 'sigma_measured': 95, 'expected': 102},
    {'T_nominal': 40, 't_nominal': 28, 'sigma_measured': 100, 'expected': 106.7},
]

print(f"\n{'标称条件':<20} {'实测强度':<12} {'预期强度':<12} {'实际Ψ':<12} {'等效龄期':<12}")
print("-" * 78)

for case in measured_cases:
    T_nom, t_nom = case['T_nominal'], case['t_nominal']
    sigma_m = case['sigma_measured']
    sigma_e = case['expected']
    
    # 计算实测对应的Ψ
    phi_actual = sigma_m / sigma_max
    
    # 反解Ψ
    def equation_psi(psi):
        return phi_eml(psi) - phi_actual
    
    psi_actual = fsolve(equation_psi, 0.5)[0]
    
    # 假设温度准确，反推等效龄期
    H_actual = psi_actual / np.exp(-gamma * T_nom)
    if H_actual < 1:
        t_effective = -np.log(1 - H_actual) / k
    else:
        t_effective = np.inf
    
    print(f"{T_nom}℃×{t_nom}d{'':<10} {sigma_m:<12.1f} {sigma_e:<12.1f} {psi_actual:<12.3f} {t_effective:<12.1f}d")

print("\n【应用5】工艺窗口设计 - 确定合格养护范围")
print("=" * 78)

print("\n场景：确定满足设计强度要求的养护工艺窗口")
print("-" * 78)

phi_min = 0.85  # 最低要求强度比 (85%)

print(f"\n设计标准：强度比 ≥ {phi_min} ({phi_min*sigma_max:.1f} MPa)")
print("\n合格工艺窗口 (T, t)：")

# 绘制工艺窗口
T_range = np.linspace(10, 100, 91)
t_range = np.linspace(1, 56, 56)

T_grid, t_grid = np.meshgrid(T_range, t_range)
phi_grid = np.zeros_like(T_grid)

for i in range(len(t_range)):
    for j in range(len(T_range)):
        H = H_calc(t_range[i], T_range[j])
        psi = psi_calc(H, T_range[j])
        phi_grid[i, j] = phi_eml(psi)

# 找到边界
print("\n边界条件 (T, t_min)：")
for T in [20, 30, 40, 50, 60, 70]:
    t_min = None
    for t in t_range:
        H = H_calc(t, T)
        psi = psi_calc(H, T)
        if phi_eml(psi) >= phi_min:
            t_min = t
            break
    if t_min:
        print(f"  温度 {T}℃: 最小龄期 = {t_min:.0f}d")
    else:
        print(f"  温度 {T}℃: 无法在56d内达标")

print("\n【应用6】敏感性分析 - 识别关键控制参数")
print("=" * 78)

print("\n场景：优化资源投入，识别对强度影响最大的因素")
print("-" * 78)

# 基准条件
T_base, t_base = 40, 7
H_base = H_calc(t_base, T_base)
psi_base = psi_calc(H_base, T_base)
phi_base = phi_eml(psi_base)

# 计算各参数敏感性
sensitivities = {}

# 温度敏感性
delta_T = 5
H_T = H_calc(t_base, T_base + delta_T)
psi_T = psi_calc(H_T, T_base + delta_T)
phi_T = phi_eml(psi_T)
sensitivities['温度'] = abs(phi_T - phi_base) / delta_T / phi_base * 100

# 龄期敏感性
delta_t = 1
H_t = H_calc(t_base + delta_t, T_base)
psi_t = psi_calc(H_t, T_base)
phi_t = phi_eml(psi_t)
sensitivities['龄期'] = abs(phi_t - phi_base) / delta_t / phi_base * 100

# EML参数敏感性 (假设10%变化)
for param_name, param_idx, delta in [('a1', 0, 0.018), ('b1', 1, 0.324), 
                                       ('a2', 2, 0.001), ('b2', 3, 0.1)]:
    params = [a1, b1, a2, b2]
    params[param_idx] += delta
    phi_p = phi_eml(psi_base, *params)
    sensitivities[param_name] = abs(phi_p - phi_base) / (delta/params[param_idx]) / phi_base * 100

print(f"\n在基准条件 (40℃, 7d) 下：")
print(f"基准强度比 Φ = {phi_base:.3f}")
print(f"\n参数敏感性（每1%参数变化引起的强度变化%）：")
for param, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {sens:.2f}%")

print("\n" + "=" * 78)
print("总结：EML公式的6大实际应用场景")
print("=" * 78)
print("""
1. 性能预测 - 给定(T,t)预测强度
2. 最优策略 - 求最大强度的养护条件  
3. 配方优化 - 标定新原材料的EML参数
4. 质量控制 - 反推实际养护效果
5. 工艺窗口 - 确定合格养护范围
6. 敏感性分析 - 识别关键控制参数

统一优势：所有应用基于单一算子 Φ = eml(eml(·), eml(·))
""")
