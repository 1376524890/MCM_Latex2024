import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# 数据准备
data = {
    "Year": [1995, 1998, 2002, 2006, 2019, 2022, 2023],
    "Passengers": [380600, 568500, 741500, 951400, 1305700, 1167000, 1650000],
    "% Change": [np.nan, 49, 30, 28, 37, -11, 41]
}
df = pd.DataFrame(data)
index = np.arange(len(df))

# 创建画布
fig, ax1 = plt.subplots(figsize=(20, 12), facecolor='white')  # 增大画布尺寸
plt.style.use('seaborn-v0_8-whitegrid')

# ==================== 背景处理 ====================
img = Image.open("figures\Mendenhall_Glacier.jpg").convert('RGBA')
img_arr = np.array(img)

x_min, x_max = -0.5, len(df)-0.5
y_min, y_max = df["Passengers"].min()*0.8, df["Passengers"].max()*1.2  # 扩展y轴范围

ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)

ax1.imshow(img_arr, 
          extent=[x_min, x_max, y_min, y_max],
          aspect='auto', 
          alpha=0.25,
          interpolation='lanczos',
          zorder=0)

# ==================== 颜色定义 ==================== 
cmap = plt.get_cmap('Blues')
colors = [cmap(i/len(df)) for i in range(1, len(df)+1)]

# ==================== 主柱状图美化 ====================
bars = ax1.bar(index, df["Passengers"], 
              color=colors,
              edgecolor='#1f3b4d',
              linewidth=2,
              width=0.7, 
              alpha=0.9,
              zorder=3)

# ==================== 坐标轴美化 ====================
# X轴设置
ax1.set_xticks(index)
ax1.set_xticklabels(df["Year"], 
                   fontname='Arial', 
                   fontsize=22,  # 年份字号加大
                   color='#2c3e50')
ax1.set_xlabel('Year',  # 新增x轴标签
              fontname='Arial',
              fontsize=24,
              labelpad=15,
              color='#2c3e50')

# 左侧Y轴
ax1.set_ylabel('Cruise Passenger Volume', 
              fontname='Arial', 
              fontsize=24,
              labelpad=25,
              color='#2c3e50')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1000:,.0f}K"))
ax1.tick_params(axis='y', 
               colors='#2c3e50', 
               labelsize=20,  # 刻度字号加大
               length=8,
               width=1.5)

# 右侧Y轴
ax2 = ax1.twinx()
line = ax2.plot(index, df["% Change"], 
               color='#e74c3c',
               marker='D',
               markersize=18,  # 标记尺寸加大
               linewidth=4,
               linestyle='--',
               alpha=0.9,
               zorder=3)

ax2.set_ylabel('Annual Change (%)', 
              fontname='Arial', 
              fontsize=24,
              labelpad=25,
              color='#e74c3c')
ax2.set_ylim(-20, 80)
ax2.tick_params(axis='y', 
               colors='#e74c3c', 
               labelsize=20,
               length=8,
               width=1.5)

# ==================== 数据标签美化 ====================
# 柱状图数据标签
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.01,
            f'{height/1000:,.0f}K',
            ha='center', 
            va='bottom', 
            fontsize=20,  # 加大到20
            fontweight='bold',
            color='#2c3e50',
            zorder=4)

# # 百分比变化标签
# for i, y in zip(index, df["% Change"]):
#     if not np.isnan(y):
#         ax2.text(i, y+3, f'{int(y)}%',  # 调整垂直位置
#                 ha='center', 
#                 va='bottom', 
#                 color='#e74c3c', 
#                 fontsize=22,  # 加大到22
#                 fontweight='bold',
#                 bbox=dict(facecolor='white', 
#                          edgecolor='#e74c3c', 
#                          boxstyle='round,pad=0.3',
#                          alpha=0.9),
#                 zorder=4)

# ==================== 标题美化 ====================
plt.title('Juneau Cruise Passenger Volumes (1995-2023)\n',
         fontsize=28,  # 标题加大
         fontweight='bold',
         color='#2c3e50',
         pad=35,
         loc='left')

# ==================== 图例美化 ====================
legend_elements = [
    plt.Line2D([0], [0], 
              color='#e74c3c', 
              lw=4, 
              marker='D', 
              markersize=20,  # 图例标记加大
              label='Annual Change'),
    plt.Rectangle((0,0),1,1, 
                 color=colors[3], 
                 ec='#1f3b4d', 
                 label='Passenger Volume')
]

ax1.legend(handles=legend_elements, 
          loc='upper center', 
          bbox_to_anchor=(0.5, -0.18),  # 调整位置
          ncol=2,
          frameon=True,
          framealpha=0.95,
          fontsize=22,  # 图例文字加大
          title_fontsize='24', 
          edgecolor='#bdc3c7',
          borderpad=1.5)

# ==================== 布局调整 ====================
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)  # 增加底部边距
plt.savefig('cruise_volume.png', dpi=300, bbox_inches='tight')
plt.show()