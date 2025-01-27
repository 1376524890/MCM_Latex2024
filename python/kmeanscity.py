import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ====================== Initialization ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# ====================== Data Loading & Cleaning ======================
df = pd.read_excel('data/聚类数据.xlsx', sheet_name='Sheet1')

# Column standardization
df.columns = [
    'year', 'city', 'gdp', 'real_income', 
    'nominal_income', 'gdp_growth', 'nominal_income_growth'
]

# ====================== Feature Engineering ======================
def calculate_growth(series):
    """Calculate compound growth rate"""
    if len(series) < 2:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1/(len(series)-1)) - 1

# Feature aggregation
aggregated = df.groupby('city').agg({
    'gdp': ['mean', calculate_growth],
    'real_income': ['mean', calculate_growth],
    'nominal_income': ['mean', calculate_growth],
    'gdp_growth': 'mean',
    'nominal_income_growth': 'mean'
})

# Flatten column names
aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
aggregated = aggregated.dropna()

# ====================== Data Standardization ======================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(aggregated)

# ====================== Elbow Method Visualization ======================
plt.figure(figsize=(10, 6), dpi=100)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='#2c7fb8')
plt.title('Elbow Method - Optimal Cluster Selection', fontsize=14, pad=20)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.xticks(range(1, 11))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# ====================== Clustering Execution ======================
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# ====================== Enhanced PCA Visualization ======================
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# 创建网格和预测背景
h = 0.02  # 网格步长
x_min, x_max = principal_components[:, 0].min() - 1, principal_components[:, 0].max() + 1
y_min, y_max = principal_components[:, 1].min() - 1, principal_components[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 反向转换到原始特征空间进行预测
mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
Z = kmeans.predict(mesh_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8), dpi=100)

# 绘制背景区域（离散配色）
plt.contourf(xx, yy, Z, 
            alpha=1, 
            levels=np.arange(-0.5, 3.5), 
            colors=['#F0F0FF', '#F0FFF0', '#FFF0F0'])

# 绘制数据点（渐变配色）
scatter = plt.scatter(
    principal_components[:, 0], 
    principal_components[:, 1],
    c=clusters,  # 保持使用聚类标签
    cmap='viridis',  # 使用连续渐变色系
    s=100,
    edgecolor='w',
    alpha=0.8,
    zorder=3  # 确保点在顶层
)

# 绘制聚类中心
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    s=300,
    marker='*',
    c='gold',
    edgecolor='black',
    label='Cluster Centers',
    zorder=4
)

# 添加城市标签
for i, city in enumerate(aggregated.index):
    plt.text(
        principal_components[i, 0]+0.1, 
        principal_components[i, 1]+0.1, 
        city,
        fontsize=9,
        alpha=0.75,
        zorder=5
    )

plt.title('City Clustering - PCA Projection', fontsize=14, pad=20)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Intensity', rotation=270, labelpad=15)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ====================== Enhanced Results Output ======================
aggregated['Cluster'] = clusters

# Cluster centers analysis
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=aggregated.columns[:-1]
)
cluster_centers['Cluster'] = range(kmeans.n_clusters)

print("="*60)
print("Cluster Centers Characteristics:")
print(cluster_centers.round(2))

print("\n" + "="*60)
print("Detailed Clustering Results:")
aggregated_sorted = aggregated.sort_values(by='Cluster')
print(aggregated_sorted.round(2))

# Save results
aggregated_sorted.to_excel('cluster_analysis_results.xlsx', float_format="%.2f")
print("\n" + "="*60)
print("Results saved to: cluster_analysis_results.xlsx")