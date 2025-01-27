import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 新数据集 (2010-2030)
data = {
    'year': [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
             2022,2023,2024,2025,2026,2027,2028,2029,2030],
    'Tourist Volume': [1532400,1556800,1586000,1693800,1659600,1780000,1857500,1926300,2026300,2213000,
                       2001402,2760678,2534775,2632950,2734927,2840854,2950884,3065175,3183893],
    't': [0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20],
    'Tourist Volume (Million)': [1.5324,1.5568,1.586,1.6938,1.6596,1.78,1.8575,1.9263,2.0263,2.213,
                                 2.001402,2.760678,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
    'Tourist Volume Forecast (Million)': [1.489,1.54667,1.606575,1.6688,1.733435,1.800573,1.870311,1.942751,
                                          2.017996,2.096155,2.34927,2.44026,2.534775,2.63295,2.734927,2.840854,
                                          2.950884,3.065175,3.183893],
    'Tourist Volume Forecast': [1489000,1546671,1606575,1668800,1733435,1800573,1870311,1942751,2017996,2096155,
                                2349270,2440260,2534775,2632950,2734927,2840854,2950884,3065175,3183893],
    'Carbon Emission Forecast': [121196,104449,89693,77191,67235,60142,56257,55961,59666,67824,
                                 124163,155517,194267,241166,297035,362765,439325,527770,629242],
    'Dissatisfaction Rate Forecast (%)': [19.75,18.70,17.70,16.76,15.88,15.09,14.39,13.79,13.32,12.98,
                                          12.97,13.37,14.01,14.92,16.12,17.66,19.57,21.88,24.63],
    'Revenue Forecast': [16525987,24402723,32584535,41083239,49911108,59080891,68605831,78499684,88776738,99451835,
                         134022560,146450083,159358940,172767773,186695946,201163575,216191552,231801582,248016208]
}

df = pd.DataFrame(data)

# 设置可视化样式
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 15), facecolor='#f0f0f0')
fig.suptitle('Long-term Tourism Development Forecast Analysis (2010-2030)', 
             y=1.02, fontsize=18, fontweight='bold')

# 创建颜色映射
colors = sns.color_palette("husl", 4)

# ========== 游客量对比图 ==========
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

# 仅显示到2023年的真实数据
real_data_mask = df['year'] <= 2023
ax1.plot(df[real_data_mask]['year'], df[real_data_mask]['Tourist Volume']/1e6, 
         marker='o', markersize=6, 
         linewidth=2.5, color=colors[0],
         label='Actual (Million)', alpha=0.9)

# 全周期预测数据
ax1.plot(df['year'], df['Tourist Volume Forecast (Million)'], 
         marker='s', markersize=6, 
         linewidth=2.5, color=colors[1], 
         linestyle='--', label='Forecast')

# 仅填充真实数据区间的差异
ax1.fill_between(df[real_data_mask]['year'], 
                 df[real_data_mask]['Tourist Volume']/1e6, 
                 df[real_data_mask]['Tourist Volume Forecast (Million)'], 
                 color=colors[2], alpha=0.1)

ax1.set_title('Tourist Volume Trend Comparison', pad=15, fontsize=14)
ax1.set_ylabel('Tourist Volume (Million Visitors)')
ax1.legend(frameon=True, shadow=True, loc='upper left')
ax1.grid(True, alpha=0.4)
ax1.set_xticks(np.arange(2010, 2031, 2))
ax1.set_facecolor('#fafafa')

# ========== 碳排放环形图 ==========
ax2 = plt.subplot2grid((3, 2), (1, 0))
wedges, texts = ax2.pie(df['Carbon Emission Forecast'], 
                        colors=sns.color_palette("Reds", len(df)),
                        startangle=90,
                        wedgeprops=dict(width=0.4))

ax2.set_title('Carbon Emission Distribution', y=1.08, fontsize=14)
plt.setp(texts, visible=False)
ax2.legend(wedges, df['year'],
           title="Year",
           loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1),
           fontsize=8)

# ========== 满意度-收入双轴图 ==========
ax3 = plt.subplot2grid((3, 2), (1, 1))
line = ax3.plot(df['year'], df['Dissatisfaction Rate Forecast (%)'], 
                marker='D', color=colors[2], 
                linewidth=2, label='Dissatisfaction Rate')
ax3.set_ylabel('Dissatisfaction Rate (%)', color=colors[2])
ax3.tick_params(axis='y', labelcolor=colors[2])
ax3.set_ylim(10, 26)

ax4 = ax3.twinx()
bars = ax4.bar(df['year'], df['Revenue Forecast']/1e8, 
               color=colors[3], alpha=0.6, 
               width=0.7, label='Revenue')
ax4.set_ylabel('Revenue (100 million USD)', color=colors[3])
ax4.tick_params(axis='y', labelcolor=colors[3])
ax4.set_ylim(0, 25)

ax3.set_title('Economic Indicators Relationship', fontsize=14)
ax3.set_xticks(np.arange(2010, 2031, 2))

# 添加组合图例
lines = [line[0], bars]
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left', frameon=True)

# ========== 预测误差带图 ==========
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
# 仅使用预测数据（2024年后无真实数据）
forecast_years = df['year'][df['year'] >= 2010]  # 全周期
forecast_values = df['Tourist Volume Forecast (Million)']

ax5.plot(forecast_years, forecast_values, 
        marker='o', color=colors[0],
        label='Forecast Values')

# 计算置信区间（示例数据）
lower_bound = forecast_values * 0.95
upper_bound = forecast_values * 1.05

ax5.fill_between(forecast_years, lower_bound, upper_bound, 
                color=colors[1], alpha=0.2,
                label='95% Confidence Interval')

ax5.set_title('Forecast Accuracy Analysis', fontsize=14)
ax5.set_xlabel('Year')
ax5.set_ylabel('Tourist Volume (Million)')
ax5.legend(frameon=True)
ax5.grid(True, linestyle='--', alpha=0.6)
ax5.set_xticks(np.arange(2010, 2031, 2))

plt.tight_layout()
plt.show()