import numpy as np
import math

def sigmoidActivate(num):
    return 1 / (1 + math.exp(-num))

def sigmoidActivateGradient(num):
    return sigmoidActivate(num) * (1 - sigmoidActivate(num))

class Network:
    def __init__(self, structure: list):
        self.structure = structure
        self.size = len(structure)
        self.weights = [np.random.normal(0, 1, size=(size, 1)) for size in structure[1:]]
        self.bias = [np.random.normal(0, 1, size=(structure[layer + 1], structure[layer])) for layer in range(self.size - 1)]

    def activate(self, inputs):
        current_layer_inputs = inputs
        for layer in range(self.size - 1):
            current_layer_output = sigmoidActivate(np.dot(self.weights[layer], current_layer_inputs) + self.bias[layer])
            current_layer_inputs = current_layer_output
        return current_layer_output
    
    def learn(self, training_data, learn_rate):
        weight_alterations = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights]
        bias_alterations = [np.zeros(bias_matrix.shape) for bias_matrix in self.bias]

        for training_pair in training_data:
            inputs = training_pair[0]
            expected_output = training_pair[1]

            inputs_history = np.array([inputs])
            current_layer_inputs = inputs
            for layer in range(self.size - 1):
                current_layer_output = np.dot(self.weights[layer], current_layer_inputs) + self.bias[layer]
                inputs_history.append(current_layer_output)
                current_layer_inputs = sigmoidActivate(current_layer_output)

            current_layer_cost_node_gradient = 2 * (current_layer_inputs - expected_output) * sigmoidActivateGradient(inputs_history[-1])
            for layer in range(self.size - 1):
                bias_alterations[-(layer + 1)] -= current_layer_cost_node_gradient
                weight_alterations[-(layer + 1)] -= current_layer_cost_node_gradient * inputs_history[-(layer + 2)]
                current_layer_cost_node_gradient = np.dot(self.weights[-(layer + 2)].transpose, current_layer_cost_node_gradient) * sigmoidActivateGradient(inputs_history[- (layer + 2)])

        weight_alterations *= learn_rate / len(training_data)
        bias_alterations *= learn_rate / len(training_data)
        self.weights += weight_alterations
        self.bias += bias_alterations