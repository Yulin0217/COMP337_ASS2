import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_dataset():
    """
    Load dataset from a CSV file without headers, use space as separator, and select columns 1-300

    Returns:
    numpy.ndarray: The dataset loaded from the CSV file
    """
    data = pd.read_csv('dataset', header=None, sep=' ', usecols=range(1, 301))
    return data.values


def ComputeDistance(a, b):
    """
    Calculate the Euclidean distance between two points a and b

    Parameters:
    a, b (numpy.ndarray): Two points in the dataset

    Returns:
    float: The Euclidean distance between a and b
    """
    return np.linalg.norm(a - b)

def generate_synthetic_data():
    """
    Generate synthetic data by creating a random dataset with the same shape as the original dataset.

    The function first loads the original dataset using the load_dataset function. It then uses the numpy random.rand function to generate a dataset of random numbers that has the same shape as the original dataset.

    The numpy random seed is set to 42 to ensure that the function generates the same synthetic data every time it is run.

    Returns:
    numpy.ndarray: The synthetic dataset
    """
    original_data = load_dataset()
    np.random.seed(42)
    synthetic_data = np.random.rand(*original_data.shape)
    return synthetic_data


def initialisation(x, k):
    """
    Initialize random seed and select k random indices to set initial centroids

    Parameters:
    x (numpy.ndarray): The dataset
    k (int): The number of clusters

    Returns:
    numpy.ndarray: The initial centroids
    """
    np.random.seed(42)
    indices = np.random.choice(len(x), k, replace=False)
    return x[indices]


def computeClusterRepresentatives(clusters):
    """
    Compute the mean of each cluster to find the new centroids, handle empty clusters

    Parameters:
    clusters (list of list): The current clusters

    Returns:
    numpy.ndarray: The new centroids
    """
    return np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else None for cluster in clusters])


def assignClusterIds(x, k, centroids):
    """
    Assign each data point to the nearest centroid to form clusters

    Parameters:
    x (numpy.ndarray): The dataset
    k (int): The number of clusters
    centroids (numpy.ndarray): The current centroids

    Returns:
    list of list: The new clusters
    """
    clusters = [[] for _ in range(k)]
    for point in x:
        closest = np.argmin([ComputeDistance(point, centroid) for centroid in centroids])
        clusters[closest].append(point)
    return clusters


def kMeans(x, k, maxIter=100):
    """
    Perform k-Means clustering with a maximum of 100 iterations

    Parameters:
    x (numpy.ndarray): The dataset
    k (int): The number of clusters
    maxIter (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
    list of list, numpy.ndarray: The final clusters and their centroids
    """
    centroids = initialisation(x, k)
    for _ in range(maxIter):
        clusters = assignClusterIds(x, k, centroids)
        new_centroids = computeClusterRepresentatives(clusters)
        if np.allclose(centroids, new_centroids, atol=1e-6, equal_nan=True):
            break
        centroids = new_centroids
    return clusters, centroids


def computeSilhouette(x, clusters):
    """
    This function evaluates the clustering quality by calculating the Silhouette coefficient for each data point in a dataset. Here's the step-by-step process:

    1. Distance Matrix: Initializes an N x N matrix where N is the number of data points, and populates it with the Euclidean distances between each pair of points.

    2. Silhouette Components: For each point, calculates two metrics:
   - a[i]: The mean distance of the point to all other points in its cluster, representing intra-cluster cohesion.
   - b[i]: The minimum mean distance of the point to points in any other cluster, representing nearest-cluster separation.

    3. Silhouette Score: For each point, computes its Silhouette score using the formula `(b[i] - a[i]) / max(a[i], b[i])`.

    4. Final Output: Returns the average of all individual Silhouette scores.

    Parameters:
    x (numpy.ndarray): The dataset
    clusters (list of list): The current clusters

    Returns:
    float: The average silhouette score
    """
    N = len(x)
    distMatrix = np.zeros((N, N))  # Create a full distance matrix
    for i in range(N):
        for j in range(N):
            if i != j:
                distMatrix[i][j] = ComputeDistance(x[i], x[j])
    silhouette = [0 for _ in range(N)]
    a = [0 for _ in range(N)]
    b = [math.inf for _ in range(N)]
    for i, point in enumerate(x):
        point_index = i
        for cluster_id, cluster in enumerate(clusters):
            cluster_indices = [np.where(np.all(x == point, axis=1))[0][0] for point in cluster]
            if point_index in cluster_indices:
                clusterSize = len(cluster_indices)
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[point_index, cluster_indices]) / (clusterSize - 1)
            else:
                if cluster_indices:
                    tempb = np.sum(distMatrix[point_index, cluster_indices]) / len(cluster_indices)
                    if tempb < b[i]:
                        b[i] = tempb
    for i in range(N):
        if a[i] != 0:
            silhouette[i] = (b[i] - a[i]) / max(a[i], b[i])
    return np.mean(silhouette) if len(silhouette) > 0 else 0  # Return the average silhouette score


def plotSilhouette():
    """
      Plot the silhouette scores for different values of k from 1 to 9

      Returns:
      None
      """
    x = generate_synthetic_data()
    silhouette_scores = [0.0]  # Start with k=1, silhouette score is 0 by definition
    for k in range(2, 10):
        clusters, _ = kMeans(x, k)
        score = computeSilhouette(x, [np.array(cluster) for cluster in clusters if cluster])  #
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')  # Start k from 1
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for KMeansSynthetic(Q2)')
    plt.savefig('Silhouette Score for KMeansSynthetic(Q2).png')
    plt.show()


plotSilhouette()
