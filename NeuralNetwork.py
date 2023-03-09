from Maths import Maths


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_neurons_first, hidden_neurons_second, output_neurons,
                 learning_rate, epochs):
        self.input_nodes = input_nodes
        self.first_hidden_neurons = hidden_neurons_first
        self.second_hidden_neurons = hidden_neurons_second
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Random weights
        self.weights_hidden_first = Maths.random_weights(input_nodes, hidden_neurons_first)
        self.weights_hidden_second = Maths.random_weights(hidden_neurons_first, hidden_neurons_second)
        self.weights_output = Maths.random_weights(hidden_neurons_second, output_neurons)
        # Random biases
        self.biases_hidden_first = Maths.random_biases(hidden_neurons_first)
        self.biases_hidden_second = Maths.random_biases(hidden_neurons_second)
        self.biases_output = Maths.random_biases(output_neurons)

    def feedforward_complete(self, inputs):
        weighted_sum_in_hidden_first = Maths.feedforward(inputs, self.weights_hidden_first, self.first_hidden_neurons,
                                                         self.biases_hidden_first)

        weighted_sum_in_hidden_second = Maths.feedforward(weighted_sum_in_hidden_first, self.weights_hidden_second,
                                                          self.second_hidden_neurons, self.biases_hidden_second)

        weighted_sum_in_output = Maths.feedforward(weighted_sum_in_hidden_second, self.weights_output,
                                                   self.output_neurons, self.biases_output)

        return weighted_sum_in_output

    def backpropagation_output_hidden(self, outputs_hidden, outputs_output, targets):
        errors_in_output = []
        # Error in the output layer calculation
        for neuron in self.output_neurons:
            error = outputs_output[neuron] * (1 - outputs_output[neuron]) * (targets[neuron] - outputs_output[neuron])
            errors_in_output.append(error)

            # Update weights and biases
            Maths.update_weights(self.weights_output, neuron, self.learning_rate, error, outputs_hidden)
            Maths.update_biases(self.biases_output, neuron, self.learning_rate, error)

        return errors_in_output
