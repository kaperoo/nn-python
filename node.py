# Neural network node class
import numpy as np
from functions import sigmoid

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.z = 0
        self.value = 0
        self.activations = None

    def getValue(self):
        return self.value
    
    def getWeights(self):
        return self.weights

    def feedForward(self, activations):
        self.value = 0
        self.activations = activations
        for i, act in enumerate(activations):
            self.value += act * self.weights[i]
        self.z = self.value + self.bias
        self.value = sigmoid(self.value + self.bias)
        return self.value
    
    def get_d_weights(self, error):
        return np.multiply(error, np.multiply((self.value * (1 - self.value)), self.activations))
        # for act in self.activations:
        #     yield error * (self.value * (1 - self.value)) * act
    
    def get_d_bias(self, error):
        return np.multiply(error, (self.value * (1 - self.value)))
    
    def get_d_error(self, error):
        return np.multiply(error, (self.value * (1 - self.value)) * self.weights)
        # for weight in self.weights:
        #     yield error * (self.value * (1 - self.value)) * weight

    def __repr__(self):
        return str(self.value)
        