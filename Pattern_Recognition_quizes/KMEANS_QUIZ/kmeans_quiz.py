import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.metrics import silhouette_score

data = pd.read_csv("./kmeans_quiz_data.csv")
# print(data)

# First set of starting points
kmeans = KMeans(n_clusters=3, n_init=1, init=np.array([[-4, 10], [0, 0], [4, 10]])).fit(
    data
)

# Second set of starting points
# kmeans = KMeans(n_clusters=3, n_init=1, init=np.array([[-2, 0], [2, 0], [0, 10]])).fit(
#    data
# )


# print("Cluster centers: \n", kmeans.cluster_centers_)
# print("Labels: \n", kmeans.labels_)

# Cohresion/ Inertia
print("Cohesion (Inertia): ", round(kmeans.inertia_, 2))


# Separation
separation = 0
distance = lambda x1, x2: sqrt(((x1.X1 - x2.X1) ** 2) + (x1.X2 - x2.X2) ** 2)
m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    ci = len(data.loc[kmeans.labels_ == i, :].index)
    separation += ci * distance(m, mi) ** 2
print("Separation: ", round(separation, 2))

# Silhouette Coefficient
print("Silhouette: ", round(silhouette_score(data, kmeans.labels_), 2))
