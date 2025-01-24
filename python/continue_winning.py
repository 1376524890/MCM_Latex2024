import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------------------
# 1. 数据加载与预处理
# ------------------------------
df = pd.read_csv('data/Wimbledon_featured_matches.csv')

# 初始化统计变量
current_player = None
current_streak = 0
streak_counts = defaultdict(int)
zero_streak_count = 0

# ------------------------------
# 2. 计算连续得分次数
# ------------------------------
for victor in df['point_victor']:
    if victor == current_player:
        current_streak += 1
    else:
        if current_streak == 1:
            zero_streak_count += 1
        elif current_streak > 1:
            streak_counts[current_streak - 1] += 1
        current_player = victor
        current_streak = 1

# 处理最后一段得分
if current_streak == 1:
    zero_streak_count += 1
elif current_streak > 1:
    streak_counts[current_streak - 1] += 1

# ------------------------------
# 3. 动态生成分类标签
# ------------------------------
observed_streaks = sorted([k for k in streak_counts.keys() if k >= 1])
max_streak = max(observed_streaks) if observed_streaks else 0
categories = ['0'] + [str(s) for s in observed_streaks if 1 <= s <= 3] + [str(s) for s in observed_streaks if s >= 4]

# 合并统计结果
merged_counts = {'0': zero_streak_count}
for cat in categories[1:]:
    merged_counts[cat] = streak_counts.get(int(cat), 0)

# ------------------------------
# 4. 可视化设置
# ------------------------------
# 专业配色方案
palette = {
    '0': '#E63946',    # 红色：交替得分
    '1-3': '#2A9D8F',  # 青色：短连胜
    '4+': '#264653'    # 深灰：长连胜
}

colors = []
for cat in categories:
    if cat == '0':
        colors.append(palette['0'])
    elif 1 <= int(cat) <= 3:
        colors.append(palette['1-3'])
    else:
        colors.append(palette['4+'])

# 创建画布
plt.figure(figsize=(14, 8))
bars = plt.bar(categories, 
               [merged_counts[cat] for cat in categories], 
               color=colors, 
               edgecolor='white', 
               linewidth=1.2, 
               width=0.7)


# ------------------------------
# 6. 图表美化
# ------------------------------
plt.title('Wimbledon Match: Consecutive Points Analysis', 
         fontsize=16, fontweight='bold', pad=20, color='#2B2D42')
plt.xlabel('Consecutive Points Won', fontsize=12, labelpad=10, color='#2B2D42')
plt.ylabel('Frequency', fontsize=12, labelpad=10, color='#2B2D42')

# 坐标轴优化
plt.xticks(rotation=45, ha='right', fontsize=10, color='#2B2D42')
plt.yticks(fontsize=10, color='#2B2D42')
plt.grid(axis='y', linestyle='--', alpha=0.7, color='#DEE2E6')

# 数据标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.5, 
                 f'{height}', 
                 ha='center', 
                 va='bottom',
                 fontsize=10,
                 color='#2B2D42')

# ------------------------------
# 7. 图例定位优化（右上角内部）
# ------------------------------
legend_elements = [
    plt.Rectangle((0,0),1,1, fc=palette['0'], edgecolor='white'),
    plt.Rectangle((0,0),1,1, fc=palette['1-3'], edgecolor='white'),
    plt.Rectangle((0,0),1,1, fc=palette['4+'], edgecolor='white')
]

# 关键修改：将图例定位到图表内部右上角
plt.legend(legend_elements,
           ['Alternating Points', 'Short Streaks (1-3)', 'Long Streaks (4+)'],
           loc='upper right',
           bbox_to_anchor=(0.98, 0.98),  # 紧贴右上角
           frameon=True,
           framealpha=0.9,
           edgecolor='#DEE2E6',
           fontsize=10)

# 调整边距
plt.tight_layout()
plt.subplots_adjust(top=0.85, right=0.95)  # 确保右侧空间充足
plt.show()