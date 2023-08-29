import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv('result.csv')
coordinates = data[['0', '1']].values

# Clustering using K-Means ++
num_clusters = 40
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10)
labels = kmeans.fit_predict(coordinates)
cluster_centers = kmeans.cluster_centers_

# Draw the clustering results
plt.scatter(coordinates[:, 1], coordinates[:, 0], c=labels)
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='red', marker='X')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-means++ Clustering Result')
plt.savefig('kmeans_result.png')
plt.show()
print(cluster_centers)
pd.DataFrame(np.squeeze(cluster_centers)).to_csv("xiaojizhan.csv")
