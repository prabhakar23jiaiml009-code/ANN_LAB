import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, m, n, dim, learning_rate=0.5, radius=None, epochs=1000):
        self.m, self.n, self.dim = m, n, dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.radius = radius if radius else max(m, n) / 2
        self.weights = np.random.rand(m, n, dim)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), (self.m, self.n))

    def _update_weights(self, x, bmu_index, epoch, time_constant):
        lr = self.learning_rate * np.exp(-epoch / self.epochs)
        radius = self.radius * np.exp(-epoch / time_constant)
        for i in range(self.m):
            for j in range(self.n):
                dist = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                if dist <= radius:
                    influence = np.exp(-(dist ** 2) / (2 * (radius ** 2)))
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])

    def train(self, data):
        time_constant = self.epochs / np.log(self.radius)
        for epoch in range(self.epochs):
            for x in data:
                bmu_index = self._find_bmu(x)
                self._update_weights(x, bmu_index, epoch, time_constant)
            if epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}/{self.epochs}")

    def map_vectors(self, data):
        return [self._find_bmu(x) for x in data]

if __name__ == "__main__":
    data = np.random.rand(200, 2)
    som = SelfOrganizingMap(m=10, n=10, dim=2, learning_rate=0.5, epochs=100)
    som.train(data)
    mapped = som.map_vectors(data)

    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c="blue", label="Input Data")
    for i, m in enumerate(mapped):
        plt.scatter(som.weights[m[0], m[1], 0], som.weights[m[0], m[1], 1], c="red", marker="x")
    plt.title("Self-Organizing Map (SOM)")
    plt.legend()
    plt.show()
