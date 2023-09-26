# Neural Network class
from layer import Layer
import numpy as np
from loss import MSE

class Network:
    def __init__(self, shape, loss_fn=MSE()):
        self.input_nodes = shape[0]
        self.output_nodes = shape[-1]
        self.hidden_layers = len(shape) - 2
        self.shape = shape
        self.error = None
        self.d_error = None
        self.result = None

        self.loss_fn = loss_fn
        self.network = np.array([])

        for i, counts in enumerate(shape):
            if i == 0:
                continue
            self.network = np.append(self.network, Layer(counts, shape[i-1]))
    
    def feedForward(self, inputs):
        if len(inputs) != self.input_nodes:
            raise Exception("Input length does not match input nodes")
        activations = inputs
        for layer in self.network:
            activations = layer.feedForward(activations)
        self.result = activations
        return activations
    
    def calculateError(self, y_hat):
        if self.result is None:
            raise Exception("Network has not been fed forward yet")
        self.error = self.loss_fn.calculateError(y_hat, self.result)
        self.d_error = self.loss_fn.derivativeError(y_hat, self.result)
        return self.loss_fn.calculateError(y_hat, self.result)
    
    def backPropagate(self, learning_rate):
        if self.result is None or self.error is None:
            raise Exception("Network has not been fed forward yet")

        der_errors = [self.d_error.copy()]
        new_weights = []
        new_biases = []

        for i, layer in enumerate(reversed(self.network)):
            node_e_values = np.zeros(layer.n_parents)
            for j, node in enumerate(layer.getNodes()):
                new_biases.append(node.get_d_bias(der_errors[-1][j]))
                new_weights.append(node.get_d_weights(der_errors[-1][j]))

                node_e_values = np.add(node_e_values, node.get_d_error(der_errors[-1][j]))

            der_errors.append(node_e_values)

        return (new_weights, new_biases)