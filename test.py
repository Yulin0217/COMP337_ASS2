import unittest
import numpy as np
from KMeans import load_dataset, ComputeDistance, initialisation, computeClusterRepresentatives, assignClusterIds, kMeans, computeSilhouette

class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.centroids = np.array([[1, 2], [9, 10]])

    def test_load_dataset(self):
        self.assertEqual(load_dataset().shape, self.dataset.shape)

    def test_ComputeDistance(self):
        self.assertEqual(ComputeDistance([1, 2], [4, 6]), np.linalg.norm(np.array([1, 2]) - np.array([4, 6])))

    def test_initialisation(self):
        self.assertEqual(len(initialisation(self.dataset, 2)), 2)

    def test_computeClusterRepresentatives(self):
        clusters = [[self.dataset[0], self.dataset[1]], [self.dataset[2], self.dataset[3], self.dataset[4]]]
        self.assertEqual(len(computeClusterRepresentatives(clusters)), 2)

    def test_assignClusterIds(self):
        clusters = assignClusterIds(self.dataset, 2, self.centroids)
        self.assertEqual(len(clusters), 2)

    def test_kMeans(self):
        clusters, centroids = kMeans(self.dataset, 2)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(centroids), 2)

    def test_computeSilhouette(self):
        clusters = [[self.dataset[0], self.dataset[1]], [self.dataset[2], self.dataset[3], self.dataset[4]]]
        self.assertIsInstance(computeSilhouette(self.dataset, clusters), float)

if __name__ == '__main__':
    unittest.main()