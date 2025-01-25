import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# 读取Excel数据
df = pd.read_excel("data/经济-游客数据统计.xlsx")  # 注意路径斜杠修正

# 提取特征和目标变量
X = df[['Total Passenger']]
y = df['Total Earnings']

# 线性回归建模
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Statsmodels统计信息
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

# --------------------------
# 可视化美化（英文标签+样式优化）
# --------------------------
plt.figure(figsize=(10, 6), facecolor='white')  # 设置背景为白色
plt.scatter(X, y, color='#2c7bb6', s=80, edgecolor='k', alpha=0.8, label='Actual Data')  # 调整点样式
plt.plot(X, y_pred, color='#d7191c', linewidth=3, linestyle='--', label='Regression Line')  # 调整线样式

# 坐标轴和标题
plt.xlabel('Total Passenger (Person-times)', fontsize=12, fontweight='bold')
plt.ylabel('Total Earnings (Currency Unit)', fontsize=12, fontweight='bold')
plt.title('Linear Regression: Total Earnings vs. Tourist Numbers', 
          fontsize=14, fontweight='bold', pad=20)

# 图例和网格
plt.legend(frameon=True, loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # 自动调整布局

plt.show()

# --------------------------
# 控制台输出（保留中文）
# --------------------------
print("回归模型公式：")
print(f"总收入 = {model.coef_[0]:.2f} × 游客人数 + {model.intercept_:.2f}")
print(f"R²值：{r2:.4f}")
print("\n详细统计信息：")
print(model_sm.summary())