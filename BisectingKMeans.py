import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_dataset():
    data = pd.read_csv('dataset', header=None, sep=' ', usecols=range(1, 301))
    return data.values


def ComputeDistance(a, b):
    return np.linalg.norm(a - b)


def initialisation(x, k):
    np.random.seed(42)
    indices = np.random.choice(len(x), k, replace=False)
    return x[indices]


def computeClusterRepresentatives(clusters):
    return np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else None for cluster in clusters])


def assignClusterIds(x, k, centroids):
    clusters = [[] for _ in range(k)]
    for point in x:
        closest = np.argmin([ComputeDistance(point, centroid) for centroid in centroids])
        clusters[closest].append(point)
    return clusters


def kMeans(x, k=2, maxIter=100):
    if isinstance(x, list):  # Ensure x is a NumPy array
        x = np.array(x)
    if x.size == 0:
        return [], []
    centroids = initialisation(x, k)
    for _ in range(maxIter):
        clusters = assignClusterIds(x, k, centroids)
        new_centroids = computeClusterRepresentatives(clusters)
        if np.allclose(centroids, new_centroids, atol=1e-6, equal_nan=True):
            break
        centroids = new_centroids
    return clusters, centroids




def bisectingKMeans(x, num_clusters=9):
    clusters = [x]  # Start with all points in one cluster
    while len(clusters) < num_clusters:
        # Choose the largest cluster to split
        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)
        # Convert largest_cluster to a NumPy array if it's not already
        if isinstance(largest_cluster, list):
            largest_cluster = np.array(largest_cluster)
        # Perform k-means with k=2 on the largest cluster
        new_clusters, _ = kMeans(largest_cluster, 2)
        clusters.extend(new_clusters)
    return clusters




def computeSilhouette(x, clusters):
    N = len(x)
    # Create a full distance matrix
    distMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                distMatrix[i][j] = ComputeDistance(x[i], x[j])

    silhouette = [0 for _ in range(N)]
    a = [0 for _ in range(N)]
    b = [math.inf for _ in range(N)]

    # Modify to use correct indices
    for i, point in enumerate(x):
        point_index = i  # This is the index of the point in the dataset
        for cluster_id, cluster in enumerate(clusters):
            cluster_indices = [np.where(np.all(x == point, axis=1))[0][0] for point in
                               cluster]  # Get indices of points in the cluster
            if point_index in cluster_indices:
                # Calculate a(i)
                clusterSize = len(cluster_indices)
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[point_index, cluster_indices]) / (clusterSize - 1)
            else:
                # Calculate b(i) to the nearest cluster
                if cluster_indices:
                    tempb = np.sum(distMatrix[point_index, cluster_indices]) / len(cluster_indices)
                    if tempb < b[i]:
                        b[i] = tempb

    # Calculate the silhouette score for each point
    for i in range(N):
        if a[i] != 0:
            silhouette[i] = (b[i] - a[i]) / max(a[i], b[i])

    # Return the average silhouette score
    return np.mean(silhouette) if len(silhouette) > 0 else 0


def plotSilhouette():
    x = load_dataset()
    silhouette_scores = []
    for num_clusters in range(1, 10):
        if num_clusters == 1:
            clusters = [x]  # All data in one cluster
            score = 0  # Silhouette score is not defined for one cluster
        else:
            clusters = bisectingKMeans(x, num_clusters)
            clusters = [np.array(cluster) for cluster in clusters if len(cluster) > 0]  # Ensure non-empty clusters
            score = computeSilhouette(x, clusters)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (s)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Bisecting k-Means')
    plt.savefig('silhouette_scores_bisecting.png')
    plt.show()

plotSilhouette()

