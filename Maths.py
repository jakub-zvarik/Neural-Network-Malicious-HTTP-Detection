import math
import random


# Necessary math operations for the neural network
class Maths:
    # Class variables for Gaussian Number generator
    GAUSSIAN_MEAN = 0
    GAUSSIAN_STANDARD_DEVIATION = 0.01

    # Generator of number in Gaussian distribution
    @staticmethod
    def gaussian_number():
        # Generate randon number in Gaussian distribution with mean and standard deviation (class variables)
        return random.gauss(Maths.GAUSSIAN_MEAN, Maths.GAUSSIAN_STANDARD_DEVIATION)

    @staticmethod
    def random_weights(rows, columns):
        weights = []
        for row in range(rows):
            weights_per_neuron = []
            for column in range(columns):
                weights_per_neuron.append(Maths.gaussian_number())
            weights.append(weights_per_neuron)
        return weights

    @staticmethod
    def random_biases(number_of_neurons):
        biases = []
        for neuron in range(number_of_neurons):
            biases.append(Maths.gaussian_number())
        return biases

    # Feedforward algorithm methods
    @staticmethod
    def weighted_sums(inputs, weights, number_of_neurons):
        weighted_sums = []
        for neuron in range(number_of_neurons):
            summation = 0
            for number in range(len(inputs)):
                summation += inputs[number] * weights[number][neuron]
            weighted_sums.append(summation)
        return weighted_sums

    @staticmethod
    def add_biases(weight_sums, biases):
        result = []
        for neuron in range(len(weight_sums)):
            result.append(weight_sums[neuron] + biases[neuron])

    @staticmethod
    def sigmoid(num):
        return 1 / (1 + math.exp(-num))

    @staticmethod
    def relu(num):
        return max(0, num)

    @staticmethod
    def activation_function(neurons):
        for neuron in neurons:
            Maths.relu(neuron)

    @staticmethod
    def feedforward(inputs, weights, number_of_neurons, biases):
        weighted_sums = Maths.weighted_sums(inputs, weights, number_of_neurons)
        Maths.add_biases(weighted_sums, biases)
        Maths.activation_function(weighted_sums)
        return weighted_sums
