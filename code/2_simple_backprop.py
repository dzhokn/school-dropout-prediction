import numpy as np              # Numpy is a library for numerical computing
import matplotlib.pyplot as plt # Matplotlib is a library for creating static, animated, and interactive visualizations


# TODO: delete me - Set the float formatter for numpy arrays
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# X1 = years of experience
X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
# X2 = level of education
X2 = [2, 5, 3, 5, 3, 4, 2, 3, 4, 4, 3, 7, 5, 6, 5, 5, 2, 3, 4, 5, 6, 7, 5, 3, 2, 4, 5, 7, 3, 5, 7, 7, 5]
# Y = salary
Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]

# Pandas data frame works with vectorized arrays (33 arrays of 1 element each)
vectorized_X1 = np.array(X1).reshape(-1, 1)
vectorized_X2 = np.array(X2).reshape(-1, 1)
Y_train = np.array(Y).reshape(-1, 1) / 20000  # We divide by 20000 to scale down the output (it must be between 0 and 1)

# Build the data frame
X_train = np.concatenate([vectorized_X1, vectorized_X2], axis=1)

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights and biases using random numbers
        self.weights = [np.random.randn(hidden_size, input_size), np.random.randn(output_size, hidden_size)]
        self.biases = [np.zeros((hidden_size, 1)), np.zeros((output_size, 1))]

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X: np.ndarray) -> np.ndarray:
        return X * (1 - X)

    def forward(self, X: np.ndarray, print_values: bool = False) -> list[np.ndarray]:
        z1 = self.weights[0] @ X.T + self.biases[0]
        a1 = self.sigmoid(z1)

        z2 = self.weights[1] @ a1 + self.biases[1]
        a2 = self.sigmoid(z2)

        return [z1, z2], [a1, a2]

    def backward(self, X: np.ndarray, Y: np.ndarray, a: list[np.ndarray]):
        a1 = a[0]
        a2 = a[1]

        # Calculate the error
        error = (a2 - Y.T) ** 2
        n = len(Y)

        dc_da2 = 2 * (a2 - Y.T)                             # Derivative of the error w.r.t. the activation
        dc_dz2 = self.sigmoid_derivative(a2) * dc_da2       # Derivative of the error w.r.t. the delta of the output layer
        dc_dw2 = np.dot(dc_dz2, a1.T) / n                   # Derivative of the error w.r.t. the weights of the output layer
        dc_db2 = np.sum(dc_dz2, axis=1, keepdims=True) / n  # Derivative of the error w.r.t. the biases of the output layer

        dc_da1 = self.weights[1].T @ dc_dz2                 # Derivative of the error w.r.t. the activation of the hidden layer
        dc_dz1 = self.sigmoid_derivative(a1) * dc_da1       # Derivative of the error w.r.t. the delta of the hidden layer
        dc_dw1 = np.dot(dc_dz1, X) / n                      # Derivative of the error w.r.t. the weights of the hidden layer
        dc_db1 = np.sum(dc_dz1, axis=1, keepdims=True) / n  # Derivative of the error w.r.t. the biases of the hidden layer

        return dc_dw1, dc_dw2, dc_db1, dc_db2, error

    def update_weights_and_biases(self, dc_dw1: np.ndarray, dc_dw2: np.ndarray, dc_db1: np.ndarray, dc_db2: np.ndarray, learning_rate: float):
        self.weights[0] -= learning_rate * dc_dw1
        self.weights[1] -= learning_rate * dc_dw2
        self.biases[0] -= learning_rate * dc_db1
        self.biases[1] -= learning_rate * dc_db2
    
    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, epochs: int):
        losses = []
        for epoch in range(epochs):
            z, a = self.forward(X)

            dc_dw1, dc_dw2, dc_db1, dc_db2, error = self.backward(X, Y, a)
            losses.append(np.sum(error))

            # Update the weights and biases
            self.update_weights_and_biases(dc_dw1, dc_dw2, dc_db1, dc_db2, learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} error: {round(np.sum(error), 3)}")
        return losses

network = NeuralNetwork(2, 2, 1)  # 2 input nodes, 2 hidden nodes, 1 output node
losses = network.train(X_train, Y_train, 0.5, 10000)

# Plot training loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()