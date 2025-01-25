import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建DataFrame（直接使用用户提供的数据）
df = pd.read_excel("data/经济-游客数据统计.xlsx")

# 计算相关系数矩阵（排除年份列）
corr_matrix = df[['Total Earnings', 'Total Passenger']].corr()

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,          # 显示数值
    cmap='coolwarm',     # 红蓝渐变色
    vmin=-1, vmax=1,     # 相关系数范围
    linewidths=0.5,
    annot_kws={'size': 14}
)

# 添加标题和格式调整
plt.title('Correlation Heatmap (r = 0.82)\n', fontsize=14, pad=20)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()