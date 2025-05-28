import numpy as np
from csv import reader
import matplotlib.pyplot as plt

def load_csv(filename: str) -> list[list[str]]:
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

def encode_categorical_variables(data: list[list[str]]):
    '''Loop through the data and encode the categorical variables'''
    new_data = []
    for row in data[1:]:
        row[-1] = 0 if row[-1] == 'Dropout' else 1 if row[-1] == 'Graduate' else 2 # 2 Enrolled
        new_data.append(row)
    return new_data

def remove_outliers_iqr(data: list[list[float]], col_idx: int, k: float) -> list[list[float]]:
    '''Remove the outliers from the data'''
    column_data = [row[col_idx] for row in data]

    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    return [row for row in data if row[col_idx] >= lower_bound and row[col_idx] <= upper_bound] # Return the rows that within range

def scale_data(X: list[list[float]]) -> list[list[float]]:
    """Standardize features by removing the mean and scaling to unit variance."""
    epsilon = 1e-8  # Small constant to prevent division by zero
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + epsilon)

def load_data():
    # Load the data
    csv = load_csv('data/students_dataset.csv')
    feature_names = csv[0][:-1]

    # Remove the first row (table headers)
    data = csv[1:]

    # Encode the 'Target' column (0 - Dropout, 1 - Graduate, 2 - Enrolled)    
    data = encode_categorical_variables(data)

    # Convert all values to float
    data = [[float(value) for value in row] for row in data]

    # Filter out the `Enrolled` students (keep only `Dropout` and `Graduate`).
    data_filtered = [row for row in data if row[-1] != 2]

    # Remove outliers (apply the IQR rule only on the non-categorical columns)
    for col_idx in [35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,19,12,6]:
        data_filtered = remove_outliers_iqr(data_filtered, col_idx, 10)

    # Y is the last (target) column of the data
    y = np.array([row[-1] for row in data_filtered], dtype=int)

    # X is the data without the last column
    X = np.array([row[:-1] for row in data_filtered], dtype=np.float32)

    # Transform the data (every column value to be between 0 and 1)
    X = scale_data(X)

    return X, y, feature_names

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # For each layer, initialize the weights and biases
        self.weights = []
        self.biases = []

        # Initialize random weights and biases for the hidden layer
        self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros(hidden_size))
    
        # Initialize random weights and biases for the output layer
        self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size))
        self.biases.append(np.zeros(output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        '''Loop through the layers and apply the sigmoid function to the output of each layer'''
        activations = []
        for i in range(len(self.weights)):
            z = np.dot(x, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
            x = a
        return activations # Return the output of the last layer

    def backward(self, activations, x, y, learning_rate):
        '''Backpropagate the error through the network'''
        # One-hot vector to encode the target values
        one_hot_y = np.zeros_like(activations[1])
        for i in range(len(y)):
            if y[i] == 0: # If, dropout                
                one_hot_y[i, 0] = 1   # Then, raise Dropout flag
            else:
                one_hot_y[i, 1] = 1   # Otherwise, raise Graduate flag

        # Compute the error
        error = one_hot_y - activations[1] # Directly compare the one-hot vector with the output of the last layer
        dz = error * self.sigmoid_derivative(activations[1]) # Compute the delta for the last layer

        # First update the weights and biases for the output layer
        dw = np.dot(activations[0].T, dz)
        self.weights[1] += learning_rate * dw / len(y) # Update the weights of the output layer
        db = np.sum(dz)
        self.biases[1] += learning_rate * db / len(y) # Update the biases of the output layer

        # Then update the weights and biases for the hidden layer
        dz = np.dot(dz, self.weights[1].T) * self.sigmoid_derivative(activations[0]) # Compute the delta for the hidden layer
        dw = np.dot(x.T, dz)
        self.weights[0] += learning_rate * dw / len(y) # Update the weights of the hidden layer
        db = np.sum(dz)
        self.biases[0] += learning_rate * db / len(y) # Update the biases of the hidden layer

        # Return the error
        return error

    def train(self, x, y, epochs, learning_rate) -> list[float]:
        '''Train the neural network by executing the forward and backward passes'''
        errors = []
        for epoch in range(epochs):
            activations = self.forward(x)
            err = self.backward(activations, x, y, learning_rate)
            err_sum = abs(np.sum(err))
            errors.append(err_sum)

            # Print the error
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Error: {err_sum}")

            # Prevent overfitting
            if -0.01 < err_sum < 0.01:
                print(f"Error: {err_sum}. Stopping training.")
                break
        return errors

    def rank_features(self):
        '''Rank the features by their importance. Use the weights of the hidden layer to rank the features'''
        w = self.weights[0]
        ranked_features = []
        for i in range(len(w)):
            ranked_features.append((i, np.sum(np.abs(w[i]))))
        ranked_features.sort(key=lambda x: x[1], reverse=True)
        return ranked_features # Return the ranked features

def main():
    X, y, feature_names = load_data()
    nn = NeuralNetwork(input_size=36, hidden_size=8, output_size=2)    
    errors = nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Plot the errors
    plt.plot(errors)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

    # Rank the features by their importance (weights of the hidden layer)
    ranked_features = nn.rank_features()
    feature_importance = [ranked_features[i][1] for i in range(len(ranked_features))]
    feature_importance_names = [feature_names[ranked_features[i][0]] for i in range(len(ranked_features))]

    # Plot the ranked features
    plt.bar(feature_importance_names, feature_importance)
    plt.xticks(rotation=90, size=8)
    plt.subplots_adjust(bottom=0.5)
    plt.show()

if __name__ == "__main__":
    main()
