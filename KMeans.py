import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def load_dataset():
    # 加载数据集
    data = pd.read_csv('dataset', header=None, sep=' ', usecols=range(1, 301))
    return data.values


def ComputeDistance(a, b):
    # 计算两点之间的欧氏距离
    return np.linalg.norm(a - b)


def initialisation(x, k):
    # 随机初始化聚类中心
    indices = np.random.choice(len(x), k, replace=False)
    return x[indices]


def computeClusterRepresentatives(clusters):
    # 计算每个聚类的代表（中心）
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])


def assignClusterIds(x, k, centroids):
    # 根据最近的聚类中心为每个点分配聚类ID
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
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


def computeSilhouette(x, clusters):
    # 简化版的Silhouette系数计算
    # 注意：这是一个非常简化的实现，实际计算更复杂
    silhouette_scores = []
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            a = np.mean([ComputeDistance(point, other) for other in cluster if not np.array_equal(point, other)])
            b = np.min([np.mean([ComputeDistance(point, other) for other in clusters[other_idx]]) for other_idx in
                        range(len(clusters)) if other_idx != idx])
            silhouette_scores.append((b - a) / max(a, b))
    return np.mean(silhouette_scores)


def plotSilhouette():
    x = load_dataset()
    silhouette_scores = []
    for k in range(2, 10):  # 从2开始，因为Silhouette系数至少需要2个聚类
        clusters, _ = kMeans(x, k)
        score = computeSilhouette(x, [np.array(cluster) for cluster in clusters if cluster])  # 确保聚类不为空
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different k')
    plt.savefig('silhouette_scores.png')
    plt.show()


plotSilhouette()
