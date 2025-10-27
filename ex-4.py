import numpy as np
import matplotlib.pyplot as plt

# Define the input range
x = np.linspace(-10, 10, 1000)

# Define activation functions
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.1 * x)

def linear(x):
    return x

# Group functions, titles, and colors
functions = [step, sigmoid, tanh_fn, relu, leaky_relu, linear]
titles = ['Step', 'Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'Linear']
colors = ['orange', 'blue', 'purple', 'green', 'red', 'brown']

# Plot each activation function
plt.figure(figsize=(11, 6))
for i, (func, title, color) in enumerate(zip(functions, titles, colors), start=1):
    plt.subplot(2, 3, i)
    y = func(x)
    plt.plot(x, y, color=color, linewidth=2)
    plt.title(title, fontsize=12, weight='bold')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("Input (x)", fontsize=10)
    plt.ylabel("Output f(x)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(-1.5, 1.5 if title != 'Linear' else 10)

plt.suptitle("Activation Functions in Neural Networks", fontsize=15, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
