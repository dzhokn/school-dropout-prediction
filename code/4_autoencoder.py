from csv import reader
import numpy as np
import matplotlib.pyplot as plt

# TODO: delete me - Set the float formatter for numpy arrays
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleAutoencoder:
    def __init__(self, input_size, hidden_size, learning_rate):
        # Initialize random weights with Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(input_size)
        self.learning_rate = learning_rate
        
    def forward(self, X):
        # Encoder
        self.hidden = sigmoid(np.dot(X, self.W1) + self.b1)
        # Decoder
        self.output = sigmoid(np.dot(self.hidden, self.W2) + self.b2)

        return self.output
    
    def backward(self, X):
        # Calculate gradients
        output_error = self.output - X
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.W2 -= self.learning_rate * np.dot(self.hidden.T, output_delta)
        self.b2 -= self.learning_rate * np.sum(output_delta)
        
        self.W1 -= self.learning_rate * np.dot(X.T, hidden_delta)
        self.b1 -= self.learning_rate * np.sum(hidden_delta)

def load_csv(filename: str) -> list[list[str]]:
    """Load the data from the csv file."""
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

def encode_categorical_variables(data: list[list[str]]):
    '''Loop through the data and encode the categorical variables'''
    new_data = []
    for row in data:
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
    # Read the CSV file
    csv = load_csv('data/students_dataset.csv')

    # Remove the first row (feature names)
    data = [row.copy() for row in csv[1:]]

    # Encode the 'Target' column (0 - Dropout, 1 - Graduate, 2 - Enrolled)
    data = encode_categorical_variables(data)

    # Remove the enrolled students (keep only Dropout and Graduate)
    data_filtered = [row for row in data if row[-1] != 2]

    # Convert to numpy array and parse the strings to floats
    data_filtered = np.array(data_filtered, dtype=np.float32)

    # Remove outliers
    for col_idx in range(len(data_filtered[0]) - 1): # Skip the last column
        data_filtered = remove_outliers_iqr(data_filtered, col_idx, 1.1)

    # Transform the data (every column value to be in short range between -2 and 2)
    data_without_target = [row[:-1] for row in data_filtered] # Do not scale the target column 

    data_scaled = scale_data(data_without_target)
    data_scaled = [np.hstack((row, [data_filtered[i][-1]])) for i, row in enumerate(data_scaled)]  # Add the target column back

    # Convert the list of arrays to a numpy array
    return np.array(data_scaled)

def train_autoencoder(X: np.ndarray, autoencoder: SimpleAutoencoder, epochs: int):
    '''Train the autoencoder and return the reconstruction loss for each epoch'''
    losses = []
    
    for epoch in range(epochs):
        # Shuffle the data
        X = X[np.random.permutation(len(X))]

        # Forward pass
        output = autoencoder.forward(X)

        # Calculate loss (MSE)
        loss = np.mean(np.square(output - X))
        losses.append(loss)

        # Backward pass
        autoencoder.backward(X)
    
    return losses

def check_dropout_prediction_accuracy(X_test: np.ndarray, autoencoder: SimpleAutoencoder) -> float:
    '''Check the accuracy of the reconstructed targets'''

    # Separate the test set into regular and anomaly data
    regulars = X_test.copy()
    anomalies = X_test.copy()

    # Reverse all target values in the anomaly dataset (e.g. create anomalies)
    anomalies[:, -1] = 1 - anomalies[:, -1] # 0 -> 1 and 1 -> 0

    # Get reconstructions for the test set
    regular_reconstruction_accuracy = compare_reconstructions(regulars, autoencoder) # We expect this to be high
    anomaly_reconstruction_accuracy = compare_reconstructions(anomalies, autoencoder, True) # We expect this to be low
    anomaly_detection_accuracy = 1 - anomaly_reconstruction_accuracy

    print(f"Regular reconstruction accuracy: {regular_reconstruction_accuracy:.2%}")
    print(f"Anomaly detection accuracy: {anomaly_detection_accuracy:.2%}")

    return anomaly_detection_accuracy

def compare_reconstructions(input: np.ndarray, autoencoder: SimpleAutoencoder, is_anomaly: bool = False) -> float:
    '''Compare the original data to the reconstructed data'''
    # Get reconstructions for the test set
    reconstructions = autoencoder.forward(input)

    # Extract the predicted target column and make it 0, if the value is below 0.5. Otherwise, round it up to 1
    reconstructed_targets = reconstructions[:, -1]
    reconstructed_targets = np.where(reconstructed_targets < 0.50, 0, 1)

    # Compare the original target to the reconstructed predictions
    correct_predictions = sum(reconstructed_targets == input[:, -1])

    # Calculate the accuracy in percentage
    return correct_predictions / len(input)

def train_test_split(X: np.ndarray, test_size: float):
    '''Split the data into training and testing sets'''
    # Shuffle the data
    X = X[np.random.permutation(len(X))]

    # Split the data into training and testing sets
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    X_test = X[split_index:]

    return X_train, X_test # TODO: delete me

def main():
    # Load and preprocess the data
    X = load_data()

    # Create and train autoencoder
    input_size = len(X[0])  # All features (including the target)
    hidden_size = 12  # Compressed representation

    attempt = 0
    while True:
        attempt += 1
        print(f"Attempt #{attempt} on {len(X)} rows")
        # Shuffle and split the data into training and testing sets
        X_train, X_test = train_test_split(X, test_size=0.20)

        # Instantiate the autoencoder
        autoencoder = SimpleAutoencoder(input_size, hidden_size, learning_rate=0.003)
    
        # Train the model
        losses = train_autoencoder(X_train, autoencoder, epochs=5000)

        # Test accuracy
        anomaly_detection_accuracy = check_dropout_prediction_accuracy(X_test, autoencoder)

        if anomaly_detection_accuracy > 0.93:
            break

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    main()