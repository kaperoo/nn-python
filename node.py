# Neural network node class
import numpy as np
from functions import sigmoid

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.new_weights = np.zeros(len(weights))
        self.bias = bias
        self.value = 0
        self.activations = None

    def getValue(self):
        return self.value

    def feedForward(self, activations):
        self.activations = activations
        self.value = 0
        for i, act in enumerate(activations):
            self.value += act * self.weights[i]
        self.value = sigmoid(self.value + self.bias)
        return self.value
    
    def backPropagate(self, learning_rate, error):
        if self.activations is None:
            raise Exception("Node has not been fed forward yet")

    def __repr__(self):
        return str(self.value)
        