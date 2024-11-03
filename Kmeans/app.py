from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
import pandas as pd

df = pd.read_csv('student_clustering.csv')
X = df.iloc[:,:].values

# centroids = [(-5, -5), (5, 5)]
# cluster_std = [1, 1]
#
# X, y = make_blobs(n_samples=100, n_features=2, centers=centroids, cluster_std=cluster_std, random_state=2)
# plt.scatter(X[:, 0], X[:, 1])

km = KMeans(n_clusters=4, max_iter=20)
y_means = km.fit_predict(X)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], c='red')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], c='blue')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], c='green')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], c='yellow')

plt.show()