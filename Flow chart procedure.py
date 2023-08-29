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

# Outputs the coordinates of the cluster center point
print("The coordinates of the cluster centroid：")
for center in centers:
    print(center)

# Gets the coordinates of a specific cluster
target_cluster_label = 0  # Suppose you want to get all the coordinates of the first cluster
target_cluster_coordinates = coordinates[label == target_cluster_label]
target_cluster_center = centers[target_cluster_label]

print("The coordinates of the centroid point for specific cluster：")
print(target_cluster_center)
# Gets the coordinates of a specific cluster
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=label)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.title('K-means Clustering Result')
plt.show()

# DBSCAN clustering using standardization
scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinates)

# Clustering using DBSCAN
eps = 0.01  # Adjust distance parameters
min_samples = 1  # Adjust minimum sample number parameter
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(scaled_coordinates)

# Gets the cluster center point
unique_labels = np.unique(labels)
cluster_centers = []
for label in unique_labels:
    if label != -1:  # -1 indicates the noise point and is not included in the cluster
        cluster_centers.append(np.mean(coordinates[labels == label], axis=0))

cluster_centers = np.array(cluster_centers)

# Displays the latitude and longitude coordinates of all hot spots
print("The latitude and longitude coordinates of all the hot spots：")
for center in cluster_centers:
    print(center)
# Plot data points and clustering centers
plt.scatter(coordinates[labels != -1, 1], coordinates[labels != -1, 0], c=labels[labels != -1])  # Plot the data points and color them according to the clustering label
plt.scatter(np.array(cluster_centers)[:, 1], np.array(cluster_centers)[:, 0], marker='x', color='red', s=50)  # Draw the cluster center point


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clustering')
plt.show()

print(f"The length of the data after the exception is deleted is：{len(coordinates[labels != -1, 1])}")
# Heat map of kernel density estimation
sns.kdeplot(x=cluster_centers[:, 1], y=cluster_centers[:, 0], cmap='coolwarm', fill=True, cbar=True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Kernel Density Estimation of Latitude and Longitude')



plt.show()
#  x 和 y coordinate
x = cluster_centers[:, 1]
y = cluster_centers[:, 0]

# Perform a kernel density estimate
kde = gaussian_kde([x, y])

# Calculate the density value for each point
densities = kde.evaluate([x, y])


print("The maximum density is：",np.max(densities))
print("The minimum density is：",np.min(densities))


threshold_min = 1.67
threshold_max = 71.6


cluster_centers_new = []
center_points = cluster_centers[(densities > threshold_min) & (densities < threshold_max)]
cluster_centers_new.append(center_points)
cluster_centers_new = np.array(cluster_centers_new)
print("Number of cluster centers after deletion：",len(cluster_centers_new))


# Heat map of kernel density estimation
sns.kdeplot(x=cluster_centers[:, 1], y=cluster_centers[:, 0], cmap='coolwarm', fill=True, cbar=True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Kernel Density Estimation of Latitude and Longitude')
# Mapping hotspot centers
if len(cluster_centers_new) > 0:
    plt.scatter(cluster_centers_new[0,:, 1], cluster_centers_new[0,:, 0], marker='X', c='blue', s= 50, alpha= 0.4)

plt.show()
print(cluster_centers_new)
pd.DataFrame(np.squeeze(cluster_centers_new)).to_csv("result.csv")
pd.DataFrame(np.squeeze(centers)).to_csv("centerresult.csv")
pd.DataFrame(np.squeeze(cluster_centers)).to_csv("result2.csv")