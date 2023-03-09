import random
import math


class Maths:

    @staticmethod
    def gaussian_number():
        while True:
            result = random.gauss(0, 1)
            if -1 <= result <= 1:
                return result

    @staticmethod
    def random_weights(rows, cols):
        weights = []
        for row in range(rows):
            weights_neuron = []
            for col in range(cols):
                weights_neuron.append(Maths.gaussian_number() * 0.01)
            weights.append(weights_neuron)
        return weights

    @staticmethod
    def random_biases(number_of_neurons):
        biases = []
        for neuron in range(number_of_neurons):
            biases.append(Maths.gaussian_number() * 0.01)
        return biases

    # Feedforward algorithm methods
    @staticmethod
    def weighted_sums(inputs, weights, number_of_neurons):
        weighted_sums = []
        for neuron in range(number_of_neurons):
            summation = 0
            for number in inputs:
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
        return 1 / (1 + math.exp(- num))

    @staticmethod
    def activation_function(neurons):
        for neuron in neurons:
            Maths.sigmoid(neuron)

    @staticmethod
    def feedforward(inputs, weights, number_of_neurons, biases):
        weighted_sums = Maths.weighted_sums(inputs, weights, number_of_neurons)
        Maths.add_biases(weighted_sums, biases)
        Maths.activation_function(weighted_sums)
        return weighted_sums

    # Backpropagation support methods
    @staticmethod
    def update_weights(weights, neuron_pointer, learning_rate, error, previous_layer):
        for weight in previous_layer:
            weights[weight][neuron_pointer] += learning_rate * error * previous_layer[weight]

    @staticmethod
    def update_biases(biases, neuron_pointer, learning_rate, error):
        biases[neuron_pointer] += learning_rate * error
