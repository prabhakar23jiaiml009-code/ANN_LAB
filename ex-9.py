#Learning Vector Quantization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.array([[1.0, 1.0],
                 [1.5, 2.0],
                 [3.0, 4.0],
                 [5.0, 7.0],
                 [3.5, 5.0],
                 [4.5, 5.0],
                 [3.5, 4.5]])

labels = np.array([0, 0, 0, 1, 1, 1, 1])

prototypes = np.array([[1.0, 1.0],
                       [5.0, 7.0]])
proto_labels = np.array([0, 1])

lr = 0.1
epochs = 20

for epoch in range(epochs):
    for i, x in enumerate(data):
        dists = np.linalg.norm(prototypes - x, axis=1)
        winner = np.argmin(dists)
        if labels[i] == proto_labels[winner]:
            prototypes[winner] += lr * (x - prototypes[winner])
        else:
            prototypes[winner] -= lr * (x - prototypes[winner])
    lr *= 0.95

colors = ['blue' if l == 0 else 'red' for l in labels]
plt.scatter(data[:, 0], data[:, 1], c=colors, marker='o', label='Data')
plt.scatter(prototypes[:, 0], prototypes[:, 1], c=['blue', 'red'], marker='X', s=200, label='Prototypes')
plt.title("Learning Vector Quantization (LVQ)")
plt.legend()
plt.show()
