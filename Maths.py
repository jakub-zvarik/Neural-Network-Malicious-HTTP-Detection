import random
import numpy as np


# Math operation for the neural network
# Random weights and biases initialisation
# Model accuracy calculation
class Maths:
    # Class variables for Gaussian Number generator
    GAUSSIAN_MEAN = 0
    GAUSSIAN_STANDARD_DEVIATION = 0.01

    # Generator of number in Gaussian distribution
    # Numbers generated in accordance to set class variables (mean and standard deviation)
    @staticmethod
    def gaussian_number():
        # Generate randon numbers in Gaussian distribution
        return random.gauss(Maths.GAUSSIAN_MEAN, Maths.GAUSSIAN_STANDARD_DEVIATION)

    # Random weights generator
    # Weights are generated in Gaussian distribution
    # Resulting array converted to NumPy array to be compatible with other NumPy operations in the Neural Network
    @staticmethod
    def random_weights(rows, columns):
        weights = []
        for row in range(rows):
            weights_per_neuron = []
            for column in range(columns):
                weights_per_neuron.append(Maths.gaussian_number())
            weights.append(weights_per_neuron)
        return np.array(weights)

    # Random biases generator
    # Biases are generated in Gaussian distribution
    # Resulting array converted to NumPy array to be compatible with other NumPy operations in the Neural Network
    @staticmethod
    def random_biases(number_of_neurons):
        biases = []
        for neuron in range(number_of_neurons):
            biases.append(Maths.gaussian_number())
        return np.array(biases)

    # ReLU (Rectified Linear Unit) - Activation function - used in feedforward algorithm
    # If number <= 0 returns 0, if > 0, returns unchanged number
    @staticmethod
    def relu_activation(number):
        return np.maximum(0, number)

    # Derivative of ReLU - similar as the ReLU activation function, used in backpropagation
    # If number <= 0 returns 0, if > 0, returns 1
    @staticmethod
    def relu_derivative(number):
        # Returns true or false as 1.0 or 0.0 if number > 0
        return (number > 0).astype(float)

    # Model accuracy calculator
    # Calculates overall accuracy, false negatives and false positives (%)
    @staticmethod
    def accuracy(predictions, targets):
        correct = 0
        false_positive = 0
        false_negative = 0
        # Sum results
        for prediction, target in zip(predictions, targets):
            if prediction == target:
                correct += 1
            elif prediction == 0 and target == 1:
                false_negative += 1
            elif prediction == 1 and target == 0:
                false_positive += 1
        # Calculate percentage
        accuracy = correct / len(predictions) * 100
        false_n = false_negative / len(predictions) * 100
        false_p = false_positive / len(predictions) * 100
        # Print the information
        print(f"\nAccuracy: {accuracy:.2f}% | False negatives: {false_n:.2f}% | False positives: {false_p:.2f}%\n")
