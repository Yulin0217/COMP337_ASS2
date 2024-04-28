import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_dataset():
    data = pd.read_csv('dataset', header=None, sep=' ', usecols=range(1, 301))
    return data.values


def generate_synthetic_data():
    # 加载原始数据以获取维度和大小
    original_data = load_dataset()
    np.random.seed(42)
    # 生成具有相同维度和数量的随机数据
    synthetic_data = np.random.rand(*original_data.shape)
    return synthetic_data


def ComputeDistance(a, b):
    return np.linalg.norm(a - b)


def initialisation(x, k):
    np.random.seed(42)  # 设置随机数种子以保证每次运行结果的一致性
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


def kMeans(x, k, maxIter=100):
    centroids = initialisation(x, k)
    for _ in range(maxIter):
        clusters = assignClusterIds(x, k, centroids)
        new_centroids = computeClusterRepresentatives(clusters)
        if np.allclose(centroids, new_centroids, atol=1e-6, equal_nan=True):
            break
        centroids = new_centroids
    return clusters, centroids


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
    x = generate_synthetic_data()  # 使用合成数据而不是加载实际数据集
    silhouette_scores = [0.0]  # Start with k=1, silhouette score is 0 by definition
    for k in range(2, 10):
        clusters, _ = kMeans(x, k)
        score = computeSilhouette(x, [np.array(cluster) for cluster in clusters if cluster])  # 确保聚类不为空
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')  # Start k from 1
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different k(Q2)')
    plt.savefig('silhouette_scores.png')
    plt.show()


plotSilhouette()
