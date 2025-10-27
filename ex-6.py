# Hebbian Learning for Logic Gates
import numpy as np


print("Choose logic gate: 1 for AND, 2 for OR")
choice = int(input("Enter your choice: "))

if choice == 1:
    print("\nTraining for AND gate...\n")
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  # bipolar inputs
    Y = np.array([[1], [-1], [-1], [-1]])               # AND gate (bipolar)
elif choice == 2:
    print("\nTraining for OR gate...\n")
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  # bipolar inputs
    Y = np.array([[1], [1], [1], [-1]])                 # OR gate (bipolar)
else:
    print("Invalid choice!")
    exit()

w = np.zeros((2, 1))
b = 0


print("Training using Hebbian Learning...\n")
for i in range(len(X)):
    x = X[i].reshape(2, 1)
    y = Y[i][0]
    w = w + x * y
    b = b + y
    print(f"After sample {i+1}: w = {w.T}, b = {b}")


print("\nTesting...\n")
for i in range(len(X)):
    net = np.dot(X[i], w) + b
    output = 1 if net >= 0 else -1
    print(f"Input: {X[i]}  ->  Output: {output}  (Expected: {Y[i][0]})")

print("\nFinal Weights:", w.T)
print("Final Bias:", b)
