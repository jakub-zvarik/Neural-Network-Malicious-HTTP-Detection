from Maths import Maths
import numpy as np


# Detection engine for the proposed Web Application Firewall
# Feedforward Neural Network with 2 hidden layers
# Weights and biases are initialised with Gaussian numbers (mean 0, standard deviation 1)
# Neural Network is using ReLU as activation function
# Adam optimiser used for backpropagation with adaptive learning rate
# Training is conducted using 'mini-batch' gradient for better performance
class DetectionEngine:

    # Constructor
    def __init__(self, hidden1_neurons, hidden2_neurons, learning_rate, output_neurons=1, input_nodes=11):
        # General
        self.learning_rate = learning_rate
        # Initialise the weights with Gaussian numbers (mean of 0, standard deviation of 1)
        self.weights_h1 = Maths.random_weights(input_nodes, hidden1_neurons)
        self.weights_h2 = Maths.random_weights(hidden1_neurons, hidden2_neurons)
        self.weights_out = Maths.random_weights(hidden2_neurons, output_neurons)
        # Initialise the biases with Gaussian numbers (mean of 0, standard deviation of 1)
        self.biases_h1 = Maths.random_biases(hidden1_neurons)
        self.biases_h2 = Maths.random_biases(hidden2_neurons)
        self.biases_out = Maths.random_biases(output_neurons)
        # Variables storing progress of the feedforward algorithm
        self.feed_input_hidden1 = None
        self.activated_hidden1 = None
        self.feed_hidden1_hidden2 = None
        self.activated_hidden2 = None
        self.feed_hidden2_output = None
        # Adam optimiser - general parameters
        self.ctrl_exp_decay_rate1 = 0.9
        self.ctrl_exp_decay_rate2 = 0.999
        self.avoid_div_by_0 = 1e-8
        self.adam_iteration = 0
        # Initialise arrays to keep track of first (m) and second (v) moments of gradient descent
        # Weights gradient moments
        self.m_weights_h1, self.v_weights_h1 = np.zeros_like(self.weights_h1), np.zeros_like(self.weights_h1)
        self.m_weights_h2, self.v_weights_h2 = np.zeros_like(self.weights_h2), np.zeros_like(self.weights_h2)
        self.m_weights_out, self.v_weights_out = np.zeros_like(self.weights_out), np.zeros_like(self.weights_out)
        # Biases gradient moments
        self.m_biases_h1, self.v_biases_h1 = np.zeros_like(self.biases_h1), np.zeros_like(self.biases_h1)
        self.m_biases_h2, self.v_biases_h2 = np.zeros_like(self.biases_h2), np.zeros_like(self.biases_h2)
        self.m_biases_out, self.v_biases_out = np.zeros_like(self.biases_out), np.zeros_like(self.biases_out)

    # Feedforward algorithm pushes the data through the network performing necessary maths operations
    # Feedforward algorithm is performed on the whole dataset that is inputted in (faster solution with NumPy)
    # Returns output (evaluation) of the data by the neural network
    def feedforward(self, inputs):
        # Dot product (of input layer and first hidden layer weights), bias addition and activation
        self.feed_input_hidden1 = np.dot(inputs, self.weights_h1) + self.biases_h1
        self.activated_hidden1 = Maths.relu_activation(self.feed_input_hidden1)
        # Dot product (of first hidden and second hidden layer) wrights, bias addition and activation
        self.feed_hidden1_hidden2 = np.dot(self.activated_hidden1, self.weights_h2) + self.biases_h2
        self.activated_hidden2 = Maths.relu_activation(self.feed_hidden1_hidden2)
        # Dot product (of second hidden and output layer weights) and bias addition
        self.feed_hidden2_output = np.dot(self.activated_hidden2, self.weights_out) + self.biases_out
        return self.feed_hidden2_output

    # Backpropagation starting at the output layer
    # Calculation of the error made
    # Gradient of the error for weights
    # Gradient of the error for biases
    def backpropagation_output(self, outputs, batch_targets):
        # Errors
        errors = outputs - batch_targets
        # Gradient - weights
        gradient_weights = np.dot(self.activated_hidden2.T, errors)
        # Gradient - biases
        gradient_biases = np.sum(errors)
        return errors, gradient_weights, gradient_biases

    # Backpropagation - hidden layers (applicable on both)
    # Calculation of the error in the hidden layer
    # Gradient of the error for weights
    # Gradient of the error for biases
    def backpropagation_hidden_layers(self, prev_error, weights, output_second, output_first):
        # Errors
        errors = np.dot(prev_error, weights.T) * Maths.relu_derivative(output_second)
        # Gradient - weights
        gradient_weights = np.dot(output_first.T, errors)
        # Gradient - biases
        gradient_biases = np.sum(errors)
        return errors, gradient_weights, gradient_biases

    # Adam (Adaptive Moment Estimation) optimiser
    # Adapt learning rate for more effective learning
    def adam_optimiser_updater(self, parameters, gradients, m_gradients, v_gradients):
        # Perform operation on all arrays
        for index in range(len(parameters)):
            # Update biased first and second moment estimates
            m_gradients[index] = self.ctrl_exp_decay_rate1 * m_gradients[index] + (1 - self.ctrl_exp_decay_rate1) * \
                                 gradients[index]
            v_gradients[index] = self.ctrl_exp_decay_rate2 * v_gradients[index] + (1 - self.ctrl_exp_decay_rate2) * (
                    gradients[index] ** 2)
            # Calculation of the bias-corrected moment estimates
            m_bias_corrected = m_gradients[index] / (1 - self.ctrl_exp_decay_rate1 ** self.adam_iteration)
            v_bias_corrected = v_gradients[index] / (1 - self.ctrl_exp_decay_rate2 ** self.adam_iteration)
            # Update parameters with adapted learning rate
            parameters[index] -= self.learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.avoid_div_by_0)
        # Update Adam optimiser parameters
        self.m_weights_h1, self.m_biases_h1, self.m_weights_h2, self.m_biases_h2, self.m_weights_out, self.m_biases_out = m_gradients
        self.v_weights_h1, self.v_biases_h1, self.v_weights_h2, self.v_biases_h2, self.v_weights_out, self.v_biases_out = v_gradients

    # Complete backpropagation process
    # Calculate errors and gradients in all layers
    # Use Adam optimiser to update all weights and biases, using information from gradient to adapt learning rate
    def backpropagation(self, batch, batch_targets, output):
        # Calculate errors and gradients for weights and biases in the output layer
        error, gradient_weights_out, gradient_biases_out = self.backpropagation_output(output, batch_targets)
        # Calculate errors and gradients for weights and biases in the second hidden layer
        gradient_h2, gradient_weights_h2, gradient_biases_h2 = self.backpropagation_hidden_layers(error,
                                                                                                  self.weights_out,
                                                                                                  self.activated_hidden2,
                                                                                                  self.activated_hidden1)
        # Calculate errors and gradients for weights and biases in the first hidden layer
        gradient_h1, gradient_weights_h1, gradient_biases_h1 = self.backpropagation_hidden_layers(gradient_h2,
                                                                                                  self.weights_h2,
                                                                                                  self.activated_hidden1,
                                                                                                  batch)
        # Create lists of parameters, gradients, first moments (m) and second moments (v) of gradient descent
        parameters = [self.weights_h1, self.biases_h1, self.weights_h2, self.biases_h2, self.weights_out,
                      self.biases_out]
        gradients = [gradient_weights_h1, gradient_biases_h1, gradient_weights_h2, gradient_biases_h2,
                     gradient_weights_out, gradient_biases_out]
        m_gradients = [self.m_weights_h1, self.m_biases_h1, self.m_weights_h2, self.m_biases_h2, self.m_weights_out,
                       self.m_biases_out]
        v_gradients = [self.v_weights_h1, self.v_biases_h1, self.v_weights_h2, self.v_biases_h2, self.v_weights_out,
                       self.v_biases_out]
        # Update weights and biases with Adam optimiser
        self.adam_optimiser_updater(parameters, gradients, m_gradients, v_gradients)

    # Training
    # Mini-Batch method
    # Random permutations to shuffle datasets to improve learning
    def train(self, dataset, targets, epochs, batch_size):
        # Rows
        number_of_rows = dataset.shape[0]
        # Rounded number of batches
        number_of_batches = number_of_rows // batch_size
        # Training
        for epoch in range(1, epochs + 1):
            # Random permutation of the dataset and targets to shuffle them
            permutation = np.random.permutation(number_of_rows)
            dataset_shuffled = dataset[permutation]
            targets_shuffled = targets[permutation]
            # Perform mini-batch training
            for batch in range(number_of_batches):
                # Calculate starting and ending point of each batch
                start = batch * batch_size
                end = min(start + batch_size, number_of_rows)
                data_batch = dataset_shuffled[start:end]
                targets_batch = targets_shuffled[start:end]
                # Feedforward the data
                output = self.feedforward(data_batch)
                # Backpropagation
                self.adam_iteration += 1
                self.backpropagation(data_batch, targets_batch, output)
            # Calculate and print error every 10 epochs
            if epoch % 10 == 0:
                # Calculate average error (squared mean of targets - outputs)
                error = np.mean((targets - self.feedforward(dataset)) ** 2)
                print(f'Epoch {epoch} | Error: {error:.4f}')

    # Prediction is used to validate neural network's training
    # Dataset that has not been seen by the neural network previously is fed into neural network for classification
    # Activation threshold is set - if the threshold is reach in any prediction, the output neuron returns true (1)
    def predict(self, dataset, targets, threshold=0.5):
        # Feedforward dataset through the neural network and return results
        output = self.feedforward(dataset)
        # Apply threshold to results and categorise them as 1 (true) if output >= threshold or 0 (false)
        predictions = (output >= threshold).astype(int)
        # Calculate model's accuracy
        Maths.accuracy(predictions, targets)
