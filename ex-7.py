import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class LogicGateNN:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, lr=0.1, epochs=10000):
        np.random.seed(42)
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.uniform(size=(input_size, hidden_size))
        self.b1 = np.random.uniform(size=(1, hidden_size))
        self.W2 = np.random.uniform(size=(hidden_size, output_size))
        self.b2 = np.random.uniform(size=(1, output_size))

    def train(self, X, y):
        for epoch in range(self.epochs):
            hidden_input = np.dot(X, self.W1) + self.b1
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.W2) + self.b2
            final_output = sigmoid(final_input)

            error = y - final_output
            d_output = error * sigmoid_derivative(final_output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_output)

            self.W2 += hidden_output.T.dot(d_output) * self.lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_hidden) * self.lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(np.square(error)):.4f}")

        return final_output

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    gate = input("Enter gate (AND / OR / XOR): ").strip().upper()
    gate_outputs = {
        "AND": [[0], [0], [0], [1]],
        "OR": [[0], [1], [1], [1]],
        "XOR": [[0], [1], [1], [0]]
    }

    if gate not in gate_outputs:
        print("Invalid gate! Defaulting to XOR.")
        gate = "XOR"

    y = np.array(gate_outputs[gate])
    model = LogicGateNN()
    predictions = model.train(X, y)

    print(f"\nFinal Predictions for {gate} gate:")
    print(np.round(predictions, 3))
