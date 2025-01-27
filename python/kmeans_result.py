import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 手动创建DataFrame
data = {
    'No.': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'Visitor Volume': [2256779, 2530518, 2612080, 2392205, 2556476, 2359848, 
                      2640759, 2449582, 2328394, 2477035, 2503671, 2584167, 2421627],
    'T Aenvironment (VC)': [50.78, 61.40, 56.06, 70.76, 59.44, 71.67, 54.34, 66.88, 
                           76.06, 64.55, 63.05, 57.35, 68.89],
    'Total Revenue (million)': [121.39, 158.78, 169.92, 139.89, 162.32, 135.47, 
                               173.83, 147.72, 131.17, 151.47, 155.11, 166.10, 143.91],
    'Carbon Emissions': [52261.30, 155696.77, 200233.36, 91139.11, 169469.73, 78586.32, 
                        216959.49, 116217.75, 65681.00, 129288.21, 142148.55, 184714.19, 103644.60]
}

df = pd.DataFrame(data)

# 选择特征并标准化（排除No.和Carbon Emissions列）
features = df.iloc[:, 1:-1]  # Visitor Volume, T Aenvironment (VC), Total Revenue
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 使用肘部法则确定最佳簇数
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 根据肘部法则选择簇数（假设选择3个簇）
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# 可视化聚类结果（PCA降维）
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Cluster Visualization (PCA Reduced)')
plt.show()

# 输出聚类结果
print("聚类结果：")
print(df[['No.', 'Visitor Volume', 'Total Revenue (million)', 'Carbon Emissions', 'Cluster']])