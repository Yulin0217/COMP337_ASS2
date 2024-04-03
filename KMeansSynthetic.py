import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filepath):
    # 尝试使用空格作为分隔符加载数据
    # 根据您的文件实际情况，可能需要调整sep参数
    data = pd.read_csv(filepath, sep=' ', header=None,usecols=range(1, 301)).values
    return data


def kMeans(x, k, maxIter=100):
    np.random.seed(42)  # 确保初始化的可复现性
    centroids = x[np.random.choice(len(x), k, replace=False)]
    for _ in range(maxIter):
        clusters = [[] for _ in range(k)]
        for point in x:
            distances = np.linalg.norm(point - centroids, axis=1)
            closest = np.argmin(distances)
            clusters[closest].append(point)
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    # 为了兼容后续的Silhouette计算，返回聚类标签
    labels = np.zeros(len(x))
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            point_idx = np.where(np.all(x == point, axis=1))[0][0]
            labels[point_idx] = cluster_idx
    return labels


def silhouette_score_manual(X, labels):
    silhouette_scores = []
    for idx, point in enumerate(X):
        own_cluster = labels[idx]
        a = np.mean([np.linalg.norm(point - other_point) for other_idx, other_point in enumerate(X) if
                     labels[other_idx] == own_cluster and idx != other_idx]) if len(X[labels == own_cluster]) > 1 else 0
        b = np.min([np.mean([np.linalg.norm(point - other_point) for other_idx, other_point in enumerate(X) if
                             labels[other_idx] == cluster]) for cluster in set(labels) if
                    cluster != own_cluster]) if len(set(labels)) > 1 else 0
        silhouette_scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return np.mean(silhouette_scores)


def plot_silhouttee(filepath):
    original_data = load_dataset(filepath)
    n_samples, n_features = original_data.shape
    print(f"Original data has {n_samples} samples with {n_features} features each.")
    X = np.random.rand(n_samples, n_features) * 2 - 1  # 生成合成数据
    scores = []

    for k in range(2, 10):
        labels = kMeans(X, k)
        score = silhouette_score_manual(X, labels)
        scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Synthetic Data with Different k')
    plt.show()


# 使用文件名'dataset'调用函数
plot_silhouttee('dataset')
