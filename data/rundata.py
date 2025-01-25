import pandas as pd
import numpy as np

# 假设总样本量N=1000
N = 1000
np.random.seed(42)

# 生成Q1的答案
q1_dist = [0.31, 0.11, 0.46, 0.11, 0.01]
q1_labels = ['Positive', 'Negative', 'Both', 'No impact', 'Don\'t know']
q1 = np.random.choice(q1_labels, size=N, p=q1_dist)

# 生成Q2的答案（仅针对Q1选Both的样本）
q2_dist = [0.38, 0.25, 0.30, 0.07]
q2_labels = ['Positive outweighs', 'Negative outweighs', 'Neutral', 'Don\'t know']
q2 = [np.random.choice(q2_labels, p=q2_dist) if resp == 'Both' else np.nan for resp in q1]

# 生成其他变量（假设外部数据）
# 冰川消融率（假设与旅游业相关）
glacier_melt = np.random.normal(loc=0.5, scale=0.1, size=N)
# 碳排放（假设与旅游业正相关）
carbon_emission = np.random.normal(loc=10, scale=2, size=N)
# 基础设施压力（Likert量表1-5）
infra_stress = np.random.randint(1, 6, size=N)
# 居民满意度（Likert量表1-5，假设与Q1相关）
satisfaction = np.where(q1 == 'Positive', np.random.randint(4, 6, size=N),
                       np.where(q1 == 'Negative', np.random.randint(1, 3, size=N),
                               np.random.randint(2, 5, size=N)))

# 创建DataFrame
df = pd.DataFrame({
    'Tourism_Impact': q1,
    'Impact_Balance': q2,
    'Satisfaction': satisfaction,
    'Glacier_Melt': glacier_melt,
    'Carbon_Emission': carbon_emission,
    'Infrastructure_Stress': infra_stress
})

# 保存数据
df.to_csv('tourism_survey_data2.csv', index=False)