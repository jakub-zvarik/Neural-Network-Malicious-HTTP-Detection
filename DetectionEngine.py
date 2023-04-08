import numpy as np


# Detection engine for the proposed Web Application Firewall
# Feedforward Neural Network (FNN) with 2 hidden layers
# FNN is using ReLU (activation function) and Adam optimisation
# Weights are initialised with 'He initialisation' and biases are initially set to 0
# Training is conducted with using 'mini-batch'
class DetectionEngine:
    # Constructor
    def __init__(self, input_nodes, hidden1_neurons, hidden2_neurons, output_neurons, learning_rate, random_state=None):
        # Set the seed for a random number generator
        np.random.seed(random_state)
        # Initialise the weights with 'He initialisation' (every weight = gaussian_num + sqrt(2/input_nodes))
        self.weights_h1 = np.random.randn(input_nodes, hidden1_neurons) * np.sqrt(2 / input_nodes)
        self.weights_h2 = np.random.randn(hidden1_neurons, hidden2_neurons) * np.sqrt(2 / hidden1_neurons)
        self.weights_out = np.random.randn(hidden2_neurons, output_neurons) * np.sqrt(2 / hidden2_neurons)
        # Initialise biases (every bias is 0 at initialisation)
        self.biases_h1 = np.zeros((1, hidden1_neurons))
        self.biases_h2 = np.zeros((1, hidden2_neurons))
        self.biases_out = np.zeros((1, output_neurons))
        # Learning rate
        self.learning_rate = learning_rate
        # Adam optimiser - general parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.time_step = 0
        # Initialise arrays to keep track of first (m) and second (v) moments of gradient descent
        # Weights moments
        self.m_weights_h1, self.v_weights_h1 = np.zeros_like(self.weights_h1), np.zeros_like(self.weights_h1)
        self.m_weights_h2, self.v_weights_h2 = np.zeros_like(self.weights_h2), np.zeros_like(self.weights_h2)
        self.m_weights_out, self.v_weights_h3 = np.zeros_like(self.weights_out), np.zeros_like(self.weights_out)
        # Biases moments
        self.m_biases_h1, self.v_biases_h1 = np.zeros_like(self.biases_h1), np.zeros_like(self.biases_h1)
        self.m_biases_h2, self.v_biases_h2 = np.zeros_like(self.biases_h2), np.zeros_like(self.biases_h2)
        self.m_biases_out, self.v_biases_out = np.zeros_like(self.biases_out), np.zeros_like(self.biases_out)

    # ReLU (Rectified Linear Unit) - Activation function - used in feedforward algorithm
    # If number <= 0 returns 0, if > 0, returns unchanged number
    def relu_activation_function(self, num):
        return np.maximum(0, num)

    # Derivative of ReLU - similar as the ReLU activation function, used in backpropagation
    # If number <= 0 returns 0, if > 0, returns 1
    def relu_derivative(self, num):
        # Returns true or false as 1 or 0
        return (num > 0).astype(float)

    # Feedforward algorithm pushes the data through the network performing necessary maths operations
    # Feedforward algorithm is performed on the whole dataset that is inputted in (faster solution)
    # Returns output (evaluation) of the data by the neural network
    def feedforward(self, inputs):
        # Dot product (input layer and first hidden layer) and biases addition
        self.feed_input_hidden1 = np.dot(inputs, self.weights_h1) + self.biases_h1
        # Activation function (ReLU) performed on the result of dot product
        self.activated_hidden1 = self.relu_activation_function(self.feed_input_hidden1)
        # Dot product (first hidden and second hidden layer) and biases addition
        self.feed_hidden1_hidden2 = np.dot(self.activated_hidden1, self.weights_h2) + self.biases_h2
        # Activation function (ReLU) performed on the result of dot product
        self.activated_hidden2 = self.relu_activation_function(self.feed_hidden1_hidden2)
        # Dot product (second hidden and output layer) and biases addition
        self.feed_hidden2_output = np.dot(self.activated_hidden2, self.weights_out) + self.biases_out
        # Return result
        return self.feed_hidden2_output

    def backpropagation(self, X, targets, output):
        # Compute gradient for the output layer and for the weights and biases connected to the output layer
        gradient_output = output - targets
        gradient_weights_out = np.dot(self.activated_hidden2.T, gradient_output) / X.shape[0]
        gradient_biases_out = np.sum(gradient_output, axis=0, keepdims=True) / X.shape[0]
        # Compute gradient for the second hidden layer and for the weights and biases connected to the second hidden layer
        gradient_h2 = np.dot(gradient_output, self.weights_out.T) * self.relu_derivative(self.activated_hidden2)
        gradient_weights_h2 = np.dot(self.activated_hidden1.T, gradient_h2) / X.shape[0]
        gradient_biases_h2 = np.sum(gradient_h2, axis=0, keepdims=True) / X.shape[0]
        # Compute gradient for the first hidden layer and for the weights and biases connected to the first hidden layer
        gradient_h1 = np.dot(gradient_h2, self.weights_h2.T) * self.relu_derivative(self.activated_hidden1)
        gradient_weights_h1 = np.dot(X.T, gradient_h1) / X.shape[0]
        gradient_biases_h1 = np.sum(gradient_h1, axis=0, keepdims=True) / X.shape[0]
        # Update the iterator for Adam optimiser
        self.time_step += 1
        # Create lists of parameters, gradients, first moments (m) and second moments (v) of gradient descent
        parameters = [self.weights_h1, self.biases_h1, self.weights_h2, self.biases_h2, self.weights_out, self.biases_out]
        gradients = [gradient_weights_h1, gradient_biases_h1, gradient_weights_h2, gradient_biases_h2, gradient_weights_out, gradient_biases_out]
        m_gradient = [self.m_weights_h1, self.m_biases_h1, self.m_weights_h2, self.m_biases_h2, self.m_weights_out, self.m_biases_out]
        v_gradient = [self.v_weights_h1, self.v_biases_h1, self.v_weights_h2, self.v_biases_h2, self.v_weights_h3, self.v_biases_out]
        # Update weights and biases with Adam optimiser
        for index in range(len(parameters)):
            # Update biased first and second moment estimates
            m_gradient[index] = self.beta1 * m_gradient[index] + (1 - self.beta1) * gradients[index]
            v_gradient[index] = self.beta2 * v_gradient[index] + (1 - self.beta2) * (gradients[index] ** 2)
            # Compute bias-corrected first and second moment estimates
            m_hat = m_gradient[index] / (1 - self.beta1 ** self.time_step)
            v_hat = v_gradient[index] / (1 - self.beta2 ** self.time_step)
            # Update parameter
            parameters[index] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        # Update Adam optimiser parameters
        self.m_weights_h1, self.m_biases_h1, self.m_weights_h2, self.m_biases_h2, self.m_weights_out, self.m_biases_out = m_gradient
        self.v_weights_h1, self.v_biases_h1, self.v_weights_h2, self.v_biases_h2, self.v_weights_h3, self.v_biases_out = v_gradient

    def train(self, X, y, epochs, batch_size=64):
        m = X.shape[0]
        num_batches = m // batch_size
        for epoch in range(1, epochs + 1):
            # Shuffle the dataset
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Perform mini-batch gradient descent
            for batch in range(num_batches + 1):
                start = batch * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch, output)

            # Print the current epoch and loss
            if epoch % 10 == 0:
                loss = np.mean((y - self.feedforward(X)) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, dataset):
        # Feedforward dataset through the neural network and return results
        output = self.feedforward(dataset)
        # Apply threshold to results and categorise them as 1 (true) or 0 (false)
        predictions = (output >= 0.5).astype(int)
        # Return final predictions of data categorisation
        return predictions.squeeze()
