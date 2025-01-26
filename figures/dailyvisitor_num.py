import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt

# 输入数据
data = {
    'visitors': [6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 
                 15000, 16000, 17000, 18000, 19000],
    'days_exceeded': [40, 39, 37, 30, 25, 22, 15, 13, 9, 6, 4, 3, 2, 1]
}
df = pd.DataFrame(data)

# 计算累积概率和Z值
df['P'] = (365 - df['days_exceeded']) / 365
df['Z'] = norm.ppf(df['P'])

# 计算日均游客量
total_visitors = 1638902
mu = total_visitors / 365

# 准备回归数据
df['X_centered'] = df['visitors'] - mu
X = df['Z'].values.reshape(-1, 1)
y = df['X_centered'].values

# 线性回归拟合标准差
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
sigma = model.coef_[0]

# 计算最高日游客量的期望值
n = 365
ln_n = math.log(n)
sqrt_2lnn = math.sqrt(2 * ln_n)
term2 = (math.log(ln_n) + math.log(4 * math.pi)) / (2 * sqrt_2lnn)
a_n = sqrt_2lnn - term2
max_visitors = mu + sigma * a_n

# 输出结果
print(f"Fitted parameters: μ = {mu:.1f}, σ = {sigma:.1f}")
print(f"Predicted maximum daily visitors: {max_visitors:.0f}")

# 绘制分布曲线
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8')  # 修改后的样式名称

# 计算实际数据分布
visitors = df['visitors'].values
days_exceeded = df['days_exceeded'].values

# 计算各区间实际天数
days_in_bins = []
for i in range(len(days_exceeded)-1):
    days_in_bins.append(days_exceeded[i] - days_exceeded[i+1])
days_in_bins.append(days_exceeded[-1])

# 计算直方图参数
bin_width = 1000
bin_centers = visitors + bin_width/2
density = np.array(days_in_bins) / (365 * bin_width)

# 绘制直方图
plt.bar(bin_centers, density, width=bin_width*0.85,
        alpha=0.7, color='#1f77b4', edgecolor='white',
        label='Actual Density')

# 生成拟合曲线
x_min = max(0, int(mu - 4*sigma))
x_max = int(mu + 4*sigma)
x = np.linspace(x_min, x_max, 1000)
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf, color='#d62728', linewidth=2.5, 
         label='Fitted Normal Distribution')

# 添加参数标注
text_str = f'μ = {mu:.1f}\nσ = {sigma:.1f}\nMax = {max_visitors:.0f}'
plt.annotate(text_str, xy=(0.72, 0.65), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle='round', alpha=0.9, 
             facecolor='white', edgecolor='#cccccc'))

# 坐标轴美化
plt.xlabel('Number of Visitors', fontsize=13, labelpad=10)
plt.ylabel('Probability Density', fontsize=13, labelpad=10)
plt.title('Visitor Distribution Analysis\n', 
          fontsize=15, fontweight='bold', pad=20)
plt.xticks(np.arange(0, 20000, 2000), 
           labels=[f'{int(x/1000)}k' for x in np.arange(0, 20000, 2000)])
plt.xlim(left=0)

# 其他美化设置
plt.legend(frameon=True, facecolor='white', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# 新增概率密度函数图
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8')

# 生成概率密度曲线
x = np.linspace(max(0, mu-4*sigma), mu+4*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

# 绘制主曲线
plt.plot(x, pdf, color='#2ca02c', linewidth=3, 
         label=f'Normal Distribution\n(μ={mu:.1f}, σ={sigma:.1f})')

# 标记最高预测值
plt.axvline(max_visitors, color='#d62728', linestyle='--', 
            linewidth=2, label=f'Predicted Max')

# 填充置信区间
plt.fill_between(x, pdf, where=(x >= mu-sigma) & (x <= mu+sigma),
                 color='#1f77b4', alpha=0.2, label='μ±1σ (68.2%)')
plt.fill_between(x, pdf, where=(x >= mu-2*sigma) & (x <= mu+2*sigma),
                 color='#1f77b4', alpha=0.1, label='μ±2σ (95.4%)')

# 美化设置
plt.title('Probability Density Function of Daily Visitors', 
          fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Number of Visitors', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xticks(np.arange(0, 25000, 2000), 
           labels=[f'{int(x/1000)}k' for x in np.arange(0, 25000, 2000)])
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=True, facecolor='white', loc='upper right')
plt.tight_layout()
plt.show()