# Neural network node class
import numpy as np
from functions import sigmoid

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.z = 0
        self.value = 0

    def getValue(self):
        return self.value
    
    def getWeights(self):
        return self.weights

    def feedForward(self, activations):
        self.value = 0
        for i, act in enumerate(activations):
            self.value += act * self.weights[i]
        self.z = self.value + self.bias
        self.value = sigmoid(self.value + self.bias)
        return self.value
    
    def __repr__(self):
        return str(self.value)
        