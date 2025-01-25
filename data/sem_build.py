import pandas as pd
from semopy import Model
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 1. 加载数据并重命名列名
# ------------------------------
def rename_columns(col):
    """将列名中的冒号及后续内容去除（例如 'Glacier_Q1: Crowding pressure' -> 'Glacier_Q1'）"""
    return col.split(':')[0].strip()

try:
    df = pd.read_csv("tourism_impact_survey_en.csv").rename(columns=rename_columns)
    print("数据加载成功，样本量：", df.shape[0])
except FileNotFoundError:
    print("错误：找不到文件 'tourism_impact_survey_en.csv'，请检查文件路径。")
    exit()

# ------------------------------
# 2. 定义模型设定（变量名已匹配）
# ------------------------------
model_spec = '''
# ========== 测量模型 ==========
Tourist =~ Tourist_Q1 + Tourist_Q2 + Tourist_Q3
Glacier =~ Glacier_Q1 + Glacier_Q2 + Glacier_Q3
Carbon =~ Carbon_Q1 + Carbon_Q2 + Carbon_Q3
Satisfaction =~ Glacier_Q3 + Carbon_Q2

# ========== 结构模型 ==========
Glacier ~ Tourist
Carbon ~ Tourist
Satisfaction ~ Glacier + Carbon
'''

# ------------------------------
# 3. 初始化并拟合模型
# ------------------------------
model = Model(model_spec)
try:
    result = model.fit(df)
    print("\n模型拟合成功！")
except Exception as e:
    print("\n模型拟合失败，错误信息：", str(e))
    exit()

# ------------------------------
# 4. 输出结果
# ------------------------------
print("\n========== 模型拟合指标 ==========")
print(result)
print("\n========== 标准化路径系数 ==========")
print(model.inspect(std_est=True))

# ------------------------------
# 5. 可视化模型（可选）
# ------------------------------
try:
    model.plot("sem_model.png", plot_covs=True, show=True, prog='dot')
    print("\n模型图已保存为 sem_model.png")
except ImportError:
    print("\n警告：未安装 graphviz，无法生成模型图。")