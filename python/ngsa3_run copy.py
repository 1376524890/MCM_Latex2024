import numpy as np
import random
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
import pandas as pd
from pandas.plotting import parallel_coordinates

# 定义优化问题类（包含4个约束）
class TourismOptimizationProblem(Problem):
    def __init__(self):
        # 决策变量范围：[年游客量, 环保税收占比]
        xl = np.array([0, 0])
        xu = np.array([4299037, 1])
        super().__init__(
            n_var=2, 
            n_obj=2, 
            n_constr=3,
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
               - 1_198_962 * 1 * V_i_million 
               + 337_702 * (V_i_million)**2 
               - (TR_i * 0.14 * TA_alpha)/460)
        f2 = C_i  # 目标2：最小化碳排放

        # ========== 约束条件计算 ==========
        g1 = V_i - 16000*365                # 日游客量上限（冗余约束）
        daily_mean = V_i_cruise / 365       # 交通负载计算
        S_i = 14_000 + (V_i_cruise * 0.14 * (1-TA_alpha)) / 10
        g2 = (daily_mean + 2.33*4589.3) - S_i  # 交通容量约束
        DS_i = 76.62 - 57.4*V_i_million + 12.9*(V_i_million)**2
        g3 = DS_i - 15                     # 居民满意度约束

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3])

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

if len(X_feasible) == 0:
    raise RuntimeError("没有找到可行解！请检查约束条件或优化参数")

# 转换目标值为实际意义
economic_benefit = -F_feasible[:, 0] / 1e6  # 转换为百万单位（最大化）
carbon_emission = F_feasible[:, 1]          # 直接取值（最小化）

# ========== 主优化结果可视化 ==========
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
best_idx = np.argmin(economic_benefit**2 + (carbon_emission/np.max(carbon_emission))**2)
plt.scatter(economic_benefit[best_idx], carbon_emission[best_idx],
            s=400, marker="*", color='gold', edgecolor='k', 
            linewidth=1, label='Optimal Compromise')

# 添加统计信息框
stats_text = f"""Statistical Summary:
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
  μ = {X_feasible[:,0].mean()/1e6:.2f}M/yr"""

plt.text(0.05, 0.95, stats_text, transform=ax.transAxes,
         fontsize=9, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 样式增强
plt.xlabel('Economic Benefit (Million USD)', fontsize=12, fontweight='bold')
plt.ylabel('Carbon Emission (Metric Tons)', fontsize=12, fontweight='bold')
plt.title('Multi-Objective Optimization: Tourism Sustainability Trade-off\n', 
          fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.gcf().set_facecolor('#EAEAF2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ========== 新增可视化分析1：决策变量分布与关系 ==========
plt.figure(figsize=(15, 10))

# 游客量分布
plt.subplot(2, 2, 1)
plt.hist(X_feasible[:, 0]/1e6, bins=20, color='skyblue', edgecolor='k', alpha=0.8)
plt.xlabel('Annual Visitors (Million)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Visitors Distribution', fontsize=12, fontweight='bold')

# 税收占比分布
plt.subplot(2, 2, 2)
plt.hist(X_feasible[:, 1]*100, bins=20, color='lightgreen', edgecolor='k', alpha=0.8)
plt.xlabel('Tax Ratio (%)', fontweight='bold')
plt.title('Tax Ratio Distribution', fontsize=12, fontweight='bold')

# 游客量与经济收益关系
plt.subplot(2, 2, 3)
sc = plt.scatter(X_feasible[:, 0]/1e6, economic_benefit, 
                 c=carbon_emission, cmap='coolwarm', alpha=0.8)
plt.colorbar(sc, label='Carbon Emission (t)')
plt.xlabel('Annual Visitors (Million)', fontweight='bold')
plt.ylabel('Economic Benefit (M$)', fontweight='bold')
plt.title('Visitors vs. Economic Benefit', fontsize=12, fontweight='bold')

# 税收占比与碳排放关系
plt.subplot(2, 2, 4)
sc = plt.scatter(X_feasible[:, 1]*100, carbon_emission, 
                 c=X_feasible[:, 0]/1e6, cmap='viridis', alpha=0.8)
plt.colorbar(sc, label='Visitors (Million)')
plt.xlabel('Tax Ratio (%)', fontweight='bold')
plt.ylabel('Carbon Emission (t)', fontweight='bold')
plt.title('Tax Ratio vs. Carbon Emission', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ========== 新增可视化分析2：约束条件分析 ==========
plt.figure(figsize=(12, 4))
constraint_names = ['Daily Capacity', 'Transport Capacity', 'Resident Satisfaction']
colors = ['#1f77b4', '#2ca02c', '#d62728']

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(res.G[feasible_mask, i], bins=20, color=colors[i], edgecolor='k', alpha=0.8)
    plt.axvline(0, color='k', linestyle='--')
    plt.title(f'Constraint: {constraint_names[i]}', fontsize=10)
    plt.xlabel('Constraint Value')
    plt.ylabel('Frequency' if i==0 else '')
    
plt.tight_layout()
plt.suptitle('Constraint Value Distribution Analysis', y=1.02, fontweight='bold')
plt.show()

# ========== 新增可视化分析3：多维数据探索 ==========
plt.figure(figsize=(14, 8))

# 平行坐标图
df = pd.DataFrame({
    'Visitors (M)': X_feasible[:, 0]/1e6,
    'Tax Ratio (%)': X_feasible[:, 1]*100,
    'Economic Benefit': economic_benefit,
    'Carbon Emission': carbon_emission,
})

# 数据标准化
df_norm = (df - df.min()) / (df.max() - df.min())
df_norm['Category'] = pd.qcut(economic_benefit, 3, labels=['Low', 'Medium', 'High'])

plt.subplot(1, 2, 1)
parallel_coordinates(df_norm, 'Category', colormap='Set2', alpha=0.5)
plt.title('Parallel Coordinates Analysis', fontweight='bold')
plt.xlabel('Parameters', fontweight='bold')
plt.ylabel('Normalized Value', fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)

# 相关系数矩阵
plt.subplot(1, 2, 2)
corr_matrix = df.corr()
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(pad=0.01)
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Feature Correlation Matrix', fontweight='bold')

# 添加相关系数数值
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', 
                 ha='center', va='center', 
                 color='white' if abs(corr_matrix.iloc[i,j])>0.5 else 'black')

plt.tight_layout()
plt.show()

# ========== 控制台输出 ==========
print("\n可行解分析：")
print(f"总解数: {len(res.X)}")
print(f"可行解数: {sum(feasible_mask)} ({sum(feasible_mask)/len(res.X):.1%})")
print(f"最大约束违反值: {np.max(res.G)}")

print("\n推荐解参数：")
print(f"年游客量: {X_feasible[best_idx, 0]:.0f} 人/年")
print(f"环保税收占比: {X_feasible[best_idx, 1]*100:.2f}%")
print(f"经济收益: {economic_benefit[best_idx]:.2f} 百万美元")
print(f"碳排放: {carbon_emission[best_idx]:.0f} 吨")

print("\n关键洞察：")
print("1. 游客量与经济效益呈现强正相关（r=%.2f）" % corr_matrix.loc['Visitors (M)', 'Economic Benefit'])
print("2. 税收占比与碳排放呈中度负相关（r=%.2f）" % corr_matrix.loc['Tax Ratio (%)', 'Carbon Emission'])
print("3. 交通容量约束是主要限制因素（%.1f%%解违反）" % 
      ((np.sum(res.G[:,1]>0)/len(res.G))*100))