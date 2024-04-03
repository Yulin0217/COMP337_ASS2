import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filepath):
    # 加载数据集
    data = pd.read_csv(filepath, sep=' ', header=None, usecols=range(1, 301)).values  # 调整这里以匹配您的数据格式
    return data


def choose_next_center(D2, x_length):
    probabilities = D2 / D2.sum()
    cumulative_probabilities = np.cumsum(probabilities)
    r = np.random.rand()
    for j, p in enumerate(cumulative_probabilities):
        if r < p:
            return j
    return x_length - 1  # 如果通过累积概率没有找到，则返回最后一个索引

def init_centroids(x, k):
    n = x.shape[0]
    centroids_idx = np.zeros(k, dtype=int)
    centroids_idx[0] = np.random.randint(0, n)
    centroids = np.zeros((k,) + x.shape[1:])
    centroids[0] = x[centroids_idx[0]]

    for i in range(1, k):
        dist_sq = np.sum((x[:, np.newaxis, :] - centroids[np.newaxis, :i, :]) ** 2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        centroids_idx[i] = choose_next_center(min_dist_sq, n)  # 确保传入 x 的长度 n
        centroids[i] = x[centroids_idx[i]]

    return centroids




def kMeans_plusplus(x, k, maxIter=100):
    np.random.seed(42)
    centroids = init_centroids(x, k)
    for _ in range(maxIter):
        distances = np.sqrt(((x - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([x[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels


def silhouette_score_manual(X, labels):
    """手动计算Silhouette系数"""
    silhouette_scores = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        own_cluster = labels[i]
        other_clusters = labels != own_cluster

        # 计算a(i)
        a_i = np.mean(np.sqrt(((X[labels == own_cluster] - X[i]) ** 2).sum(axis=1)))

        # 计算b(i)对所有其他聚类
        b_i = np.min(
            [np.mean(np.sqrt(((X[labels == k] - X[i]) ** 2).sum(axis=1))) for k in set(labels) if k != own_cluster])

        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(silhouette_scores)


def plot_silhouttee(filepath):
    x = load_dataset(filepath)
    scores = []

    for k in range(2, 10):  # K-means++至少需要2个聚类
        labels = kMeans_plusplus(x, k)
        score = silhouette_score_manual(x, labels)
        scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different k using K-means++')
    plt.show()


# 示例调用
plot_silhouttee('dataset')
