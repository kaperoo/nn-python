# Neural Network class
from layer import Layer
import numpy as np
from loss import mse

class Network:
    def __init__(self, shape, loss_fn=mse):
        self.input_nodes = shape[0]
        self.output_nodes = shape[-1]
        self.hidden_layers = len(shape) - 2
        self.shape = shape
        self.error = None
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
        self.error = self.loss_fn(y_hat, self.result)
        return self.error
    
    def backPropagate(self, learning_rate):
        if self.result is None:
            raise Exception("Network has not been fed forward yet")

        self.error = self.calculateError(self.result)

        # TODO: need to pass error to each node and calc derivatives
        for i, layer in enumerate(reversed(self.network)):
            for j, node in enumerate(layer.getNodes()):
                pass