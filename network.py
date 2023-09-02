# Neural Network class
from layer import Layer
import numpy as np

class Network:
    def __init__(self, shape):
        self.input_nodes = shape[0]
        self.output_nodes = shape[-1]
        self.hidden_layers = len(shape) - 2
        self.shape = shape

        self.network = np.array([])

        for i, counts in enumerate(shape):
            if i == 0:
                continue
            self.network = np.append(self.network, Layer(counts, shape[i-1]))
    
    def feedForward(self, inputs):
        activations = inputs
        for layer in self.network:
            activations = layer.feedForward(activations)
        return activations
