import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

data = pd.read_csv('Paris_2015_110k.csv')
coordinates = data[['latitude', 'longitude']].values


julei = KMeans(n_clusters=20, n_init=100)
julei.fit(coordinates)
label = julei.labels_
centers = julei.cluster_centers_

# 输出聚类中心点的坐标
print("The coordinates of the cluster centroid：")
for center in centers:
    print(center)

# 获取特定簇的坐标
target_cluster_label = 0  # 假设你想获取第一个簇的所有坐标
target_cluster_coordinates = coordinates[label == target_cluster_label]
target_cluster_center = centers[target_cluster_label]

print("The coordinates of the centroid point for specific cluster：")
print(target_cluster_center)
# 绘制散点图和簇中心
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=label)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.title('K-means Clustering Result')
plt.show()

# 使用标准化进行DBSCAN聚类
scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinates)

# 使用DBSCAN进行聚类
eps = 0.01  # 调整邻域距离参数
min_samples = 1  # 调整邻域最小样本数参数
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(scaled_coordinates)

# 获取聚类中心点
unique_labels = np.unique(labels)
cluster_centers = []
for label in unique_labels:
    if label != -1:  # -1表示噪声点，不包含在聚类中
        cluster_centers.append(np.mean(coordinates[labels == label], axis=0))

cluster_centers = np.array(cluster_centers)

# 显示所有热点的经纬度坐标
print("所有热点的经纬度坐标：")
for center in cluster_centers:
    print(center)
# 绘制数据点和聚类中心点
plt.scatter(coordinates[labels != -1, 1], coordinates[labels != -1, 0], c=labels[labels != -1])  # 绘制数据点，按照聚类标签着色
plt.scatter(np.array(cluster_centers)[:, 1], np.array(cluster_centers)[:, 0], marker='x', color='red', s=50)  # 绘制聚类中心点
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering')
plt.show()

print(f"删除异常点之后的数据长度为：{len(coordinates[labels != -1, 1])}")
# 绘制核密度估计热图
sns.kdeplot(x=cluster_centers[:, 1], y=cluster_centers[:, 0], cmap='coolwarm', fill=True, cbar=True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Kernel Density Estimation of Latitude and Longitude')



plt.show()
# 提取 x 和 y 坐标
x = cluster_centers[:, 1]
y = cluster_centers[:, 0]

# 执行核密度估计
kde = gaussian_kde([x, y])

# 计算每个点的密度值
densities = kde.evaluate([x, y])

# # 打印每个点的密度值
# for density in densities:
#     print("密度：", density)
print("密度最大值为：",np.max(densities))
print("密度最小值为：",np.min(densities))

# 阈值
threshold_min = 1.67
threshold_max = 71.6


cluster_centers_new = []
center_points = cluster_centers[(densities > threshold_min) & (densities < threshold_max)]
cluster_centers_new.append(center_points)
cluster_centers_new = np.array(cluster_centers_new)
print("删除完成后的聚类中心个数：",len(cluster_centers_new))


# 绘制核密度估计热图
sns.kdeplot(x=cluster_centers[:, 1], y=cluster_centers[:, 0], cmap='coolwarm', fill=True, cbar=True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Kernel Density Estimation of Latitude and Longitude')
# 绘制热点中心
if len(cluster_centers_new) > 0:
    plt.scatter(cluster_centers_new[0,:, 1], cluster_centers_new[0,:, 0], marker='X', c='blue', s= 50, alpha= 0.4)

plt.show()
print(cluster_centers_new)
pd.DataFrame(np.squeeze(cluster_centers_new)).to_csv("result.csv")
pd.DataFrame(np.squeeze(centers)).to_csv("centerresult.csv")
pd.DataFrame(np.squeeze(cluster_centers)).to_csv("result2.csv")