import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取CSV文件
df = pd.read_csv('g:\\MCM_Latex2024\\data\\Wimbledon_featured_matches.csv')

# 初始化连胜计数器和动态字典
streak_count = 0
streaks = defaultdict(int)  # 使用 defaultdict 支持任意长度的连胜

# 遍历每一行数据，计算连胜次数
for i in range(len(df)):
    if i == 0 or df.loc[i, 'point_victor'] == df.loc[i - 1, 'point_victor']:
        streak_count += 1
    else:
        streaks[streak_count] += 1
        streak_count = 0  

# 处理最后一行的连胜
if streak_count > 0:
    streaks[streak_count] += 1

# 将结果转换为普通字典（可选）
streaks = dict(streaks)

# 提取连胜次数和对应的频数
streak_lengths = list(streaks.keys())
streak_frequencies = list(streaks.values())

# 设置全局字体样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 创建图形
plt.figure(figsize=(10, 6))

# 定义颜色映射函数
def get_color(streak_length):
    if streak_length == 0:
        return 'gray'  # 0 次连胜
    elif 1 <= streak_length <= 3:
        return 'skyblue'  # 1-3 次连胜
    else:
        return 'orange'  # 4 次以上连胜

# 绘制柱状图，并根据连胜长度设置颜色
bars = plt.bar(streak_lengths, streak_frequencies, 
               color=[get_color(n) for n in streak_lengths], 
               edgecolor='black', alpha=0.8)

# 在柱状图顶端添加数字
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
             ha='center', va='bottom', fontsize=12, color='black')

# 设置标题和标签
plt.xlabel('Winning Streak Length (n)', fontsize=14, labelpad=10)
plt.ylabel('Frequency', fontsize=14, labelpad=10)
plt.title('Frequency of n-Winning Streaks', fontsize=16, pad=20)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', edgecolor='black', label='0 Streak'),
    Patch(facecolor='skyblue', edgecolor='black', label='1-3 Streaks'),
    Patch(facecolor='orange', edgecolor='black', label='4+ Streaks')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()