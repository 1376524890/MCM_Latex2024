import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# 设置全局样式
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.constrained_layout.use': True
})

# 加载数据
data = {
    'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'Greenhouse_Gas': [71727.76, 113587.49, 95652.89, 76765.20, 
                      51892.62, 56959.98, 49644.26, 49236.73, 104822.60],
    'Tourists': [1556800, 1586000, 1693800, 1659600, 
                1780000, 1857500, 1926300, 2026300, 2213000]
}
df = pd.DataFrame(data)

# 特征工程
df['Year_offset'] = df['Year'] - 2011
df['Tourists_M'] = df['Tourists'] / 1e6
df['Tourists_squared'] = (df['Tourists_M'] ** 2).round(2)

# 准备模型数据
X = df[['Year_offset', 'Tourists_M', 'Tourists_squared']]
y = df['Greenhouse_Gas']

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)
df['Predicted'] = model.predict(X)

# 创建带子图的画布
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 1.2])

# ========== 子图1：实际与预测对比 ==========
ax1 = fig.add_subplot(gs[0, 0])
actual_line, = ax1.plot(df['Year'], df['Greenhouse_Gas'], 
                       marker='o', linestyle='-', linewidth=2.5,
                       color='#2c7bb6', markersize=8, 
                       markeredgecolor='white', label='Actual')

pred_line, = ax1.plot(df['Year'], df['Predicted'], 
                     marker='s', linestyle='--', linewidth=2.5,
                     color='#d7191c', markersize=8, 
                     markeredgecolor='white', label='Predicted')

# 添加数据标签
for year, actual, pred in zip(df['Year'], df['Greenhouse_Gas'], df['Predicted']):
    ax1.text(year, actual+2500, f'{actual/1e3:.0f}k', 
            ha='center', va='bottom', color='#2c7bb6', fontsize=9)
    ax1.text(year, pred-2500, f'{pred/1e3:.0f}k', 
            ha='center', va='top', color='#d7191c', fontsize=9)

ax1.set_title('Actual vs Predicted Greenhouse Gas Emissions (2011-2019)', 
             fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Year', labelpad=10)
ax1.set_ylabel('Emissions (Metric Tons)', labelpad=10)
ax1.legend(handles=[actual_line, pred_line], loc='upper left', frameon=True)
ax1.grid(visible=True, linestyle='--', alpha=0.6)
ax1.set_ylim(30000, 130000)
ax1.set_xticks(df['Year'])
ax1.set_xticklabels(df['Year'], rotation=45)

# ========== 子图2：残差分析 ==========
ax2 = fig.add_subplot(gs[0, 1])
residuals = df['Greenhouse_Gas'] - df['Predicted']

scatter = ax2.scatter(df['Predicted'], residuals, c=df['Year'], 
                     cmap='viridis', s=120, edgecolor='white', 
                     alpha=0.8, zorder=3)

# 保留残差统计信息
stats_text = (f'Mean Residual: {residuals.mean():.1f}\n'
             f'Std Dev: {residuals.std():.1f}\n'
             f'Max Residual: {residuals.max():.1f}\n'
             f'Min Residual: {residuals.min():.1f}')
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
        va='top', ha='left', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='grey', alpha=0.8))

ax2.axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2)

ax2.set_title('Residual Analysis with Temporal Color Coding', 
             fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel('Predicted Emissions (Metric Tons)', labelpad=10)
ax2.set_ylabel('Residuals (Metric Tons)', labelpad=10)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Year', rotation=270, labelpad=15)
ax2.grid(visible=True, linestyle=':', alpha=0.5)

# ========== 子图3：游客关系分析 ==========
ax3 = fig.add_subplot(gs[1, :])
tourist_range = np.linspace(1.3e6, 2.7e6, 500)
year_fixed = 2019 - 2011

pred_df = pd.DataFrame({
    'Year_offset': year_fixed,
    'Tourists_M': tourist_range/1e6,
    'Tourists_squared': (tourist_range/1e6)**2
})
curve_pred = model.predict(pred_df)

main_line, = ax3.plot(tourist_range, curve_pred, 
                     color='#8e44ad', linewidth=3, 
                     label='Prediction Curve (2019)')

inflection_point = 1.78e6
ax3.axvline(inflection_point, color='#e67e22', linestyle='--', 
           linewidth=2, label='Critical Point (1.78M Tourists)')

sc = ax3.scatter(df['Tourists'], df['Greenhouse_Gas'], 
                c=df['Year'], cmap='viridis', 
                s=150, edgecolor='white', linewidth=1, 
                zorder=4, label='Historical Data')

# 仅保留年份标注
for idx, row in df.iterrows():
    ax3.annotate(row['Year'], (row['Tourists'], row['Greenhouse_Gas']),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, color='#34495e')

# 删除模型方程注释框
ax3.set_title('Emission-Tourist Relationship with Temporal Context', 
             fontsize=14, pad=15, fontweight='bold')
ax3.set_xlabel('Number of Tourists', labelpad=10)
ax3.set_ylabel('Emissions (Metric Tons)', labelpad=10)
ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax3.legend(loc='upper left', frameon=True)
ax3.grid(visible=True, linestyle='--', alpha=0.4)
ax3.set_xlim(1.3e6, 2.7e6)

# 调整布局
plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.show()