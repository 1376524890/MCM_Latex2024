import numpy as np
import random
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

# 定义优化问题类（包含4个约束）
class TourismOptimizationProblem(Problem):
    def __init__(self):
        # 决策变量范围：[年游客量, 环保税收占比]
        xl = np.array([0, 0])
        xu = np.array([16000*365, 1])
        super().__init__(
            n_var=2, 
            n_obj=2, 
            n_constr=3,  # 更新为4个约束
            xl=xl, 
            xu=xu,
            type_var=np.double
        )

    def _evaluate(self, x, out, *args, **kwargs):
        V_i = x[:, 0]
        TA_alpha = x[:, 1]

        # ========== 目标函数计算 ==========
        TR_i = -186843122 + 136.581 * V_i
        f1 = -TR_i  # 目标1：最大化经济收益

        V_i_million = V_i / 1e6
        V_i_cruise = (V_i - 166878.151)/1.572
        C_i = (1_151_001 - 6_723*14 
               - 1_198_962 * V_i_million 
               + 337_702 * (V_i_million)**2 
               - (TR_i * 0.14 * TA_alpha)/460)
        f2 = C_i  # 目标2：最小化碳排放

        # ========== 约束条件计算 ==========
        g1 = V_i - 16000*365                # 日游客量上限（冗余约束）
        daily_mean = V_i_cruise / 365       # 交通负载计算
        S_i = 14_000 + (V_i_cruise * 0.14 * (1-TA_alpha)) / 100
        g2 = (daily_mean + 2.33*4589.3) - S_i  # 交通容量约束
        DS_i = 76.62 - 57.4*V_i_million + 12.9*(V_i_million)**2
        g3 = DS_i - 15                     # 居民满意度约束
        # g4 = 52257-C_i                           # 新增碳排放正约束 C_i > 0 → g4 <= 0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3])  # 包含四个约束

# ========== 算法配置 ==========
ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_partitions=12)
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs, eliminate_duplicates=True)
termination = get_termination("n_gen", 300)

# ========== 运行优化 ==========
problem = TourismOptimizationProblem()
res = minimize(problem, algorithm, termination, seed=42, verbose=True)

# ========== 结果处理 ==========
feasible_mask = np.all(res.G <= 0, axis=1)
X_feasible = res.X[feasible_mask]
F_feasible = res.F[feasible_mask]

# 转换目标值为实际意义
economic_benefit = -F_feasible[:, 0] / 1e6  # 转换为百万单位（最大化）
carbon_emission = F_feasible[:, 1]          # 直接取值（最小化）

# ========== 关键修复：确保在输出前定义best_solution ==========
if len(X_feasible) > 0:
    # 计算最优折中点（仅在存在可行解时执行）
    combined_metrics = (economic_benefit**2 + (carbon_emission/np.max(carbon_emission))**2)
    best_idx = np.argmin(combined_metrics)
    best_solution = X_feasible[best_idx]
else:
    raise RuntimeError("没有找到可行解！请检查约束条件或优化参数")

# ========== 可视化增强版 ========== 
plt.figure(figsize=(12, 7))
ax = plt.gca()

# 使用税收占比作为颜色维度，增加尺寸维度表示游客量
norm = plt.Normalize(vmin=X_feasible[:, 1].min(), vmax=X_feasible[:, 1].max())
sc = plt.scatter(
    economic_benefit, 
    carbon_emission, 
    c=X_feasible[:, 1],  # 使用税收占比作为颜色维度
    cmap='plasma', 
    s=50 + (X_feasible[:, 0] / X_feasible[:, 0].max()) * 100,  # 点大小表示游客量
    alpha=0.8,
    edgecolor='w',
    linewidth=0.5
)

# 添加颜色条
cbar = plt.colorbar(sc, pad=0.02)
cbar.set_label('Environmental Tax Ratio (%)', rotation=270, labelpad=20)

# 添加趋势线（二次多项式拟合）
z = np.polyfit(economic_benefit, carbon_emission, 2)
p = np.poly1d(z)
sorted_idx = np.argsort(economic_benefit)
plt.plot(economic_benefit[sorted_idx], p(economic_benefit[sorted_idx]), 
         "--", color="gray", lw=1.5, alpha=0.7, label='Trend Line')

# 标记最佳解
if len(X_feasible) > 0:
    plt.scatter(economic_benefit[best_idx], carbon_emission[best_idx],
                s=400, marker="*", color='gold', edgecolor='k', 
                linewidth=1, label='Optimal Compromise')

# 添加统计信息框
stats_text = f"""
Statistical Summary:
- Economic Benefit: 
  μ = {economic_benefit.mean():.1f}M 
  (Δ = {economic_benefit.max()-economic_benefit.min():.1f}M)
  
- Carbon Emission: 
  μ = {carbon_emission.mean():.0f}t 
  (Δ = {carbon_emission.max()-carbon_emission.min():.0f}t)
  
- Tax Ratio: 
  μ = {X_feasible[:,1].mean()*100:.1f}% 
  (range: {X_feasible[:,1].min()*100:.1f}%-{X_feasible[:,1].max()*100:.1f}%)
  
- Visitors: 
  μ = {X_feasible[:,0].mean()/1e6:.2f}M/yr
"""

plt.text(0.05, 0.95, stats_text, transform=ax.transAxes,
         fontsize=9, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 样式增强
plt.xlabel('Economic Benefit (Million USD)', fontsize=12, fontweight='bold')
plt.ylabel('Carbon Emission (Metric Tons)', fontsize=12, fontweight='bold')
plt.title('Multi-Objective Optimization: Tourism Sustainability Trade-off\n', 
          fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)

# 设置图表背景颜色为EAEAF2
plt.gcf().set_facecolor('#EAEAF2')

# 添加图例（移动到右下角）
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='gray', markersize=8))
labels.append('Point size: Visitor volume')
ax.legend(handles, labels, loc='lower right', frameon=True)

# 调整布局，避免重叠
plt.tight_layout(pad=2.0)  # 增加边距以避免图例与图表重叠
plt.show()

# ========== 控制台输出 ========== 
print("\n可行解分析：")
print(f"总解数: {len(res.X)}")
print(f"可行解数: {sum(feasible_mask)}")
print(f"最大约束违反值: {np.max(res.G)}")

if len(X_feasible) > 0:
    print("\n推荐解参数：")
    print(f"年游客量: {best_solution[0]:.0f} 人/年")
    print(f"环保税收占比: {best_solution[1]*100:.2f}%")
    print(f"经济收益: {economic_benefit[best_idx]:.2f} 百万元")
    print(f"碳排放: {carbon_emission[best_idx]:.2f} 吨")
else:
    print("\n警告：未找到满足所有约束的可行解！")
