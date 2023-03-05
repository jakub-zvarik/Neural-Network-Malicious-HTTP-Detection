from Maths import Maths


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_neurons, output_neurons,
                 learning_rate, epochs):
        self.input_nodes = input_nodes
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Random weights
        self.weights_to_hidden = Maths.random_weights(input_nodes, hidden_neurons)
        self.weights_to_output = Maths.random_weights(hidden_neurons, output_neurons)
        # Random biases
        self.biases_hidden = Maths.random_biases(hidden_neurons)
        self.biases_output = Maths.random_biases(output_neurons)

    def feedforward(self, inputs):
        weighted_sum_in_hidden = Maths.feedforward_algorithm(inputs, self.weights_to_hidden, self.hidden_neurons,
                                                             self.biases_hidden)
        weighted_sum_in_output = Maths.feedforward_algorithm(weighted_sum_in_hidden, self.weights_to_output,
                                                             self.output_neurons, self.biases_output)
        return weighted_sum_in_output
