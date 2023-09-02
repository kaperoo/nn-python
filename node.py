# Neural network node class
import numpy as np
from functions import sigmoid

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.value = 0

    def getValue(self):
        return self.value

    def calculate_value(self):
        self.value = 0
        for i in self.parents:
            self.value += i.getValue * self.weights[i]
        self.value += self.bias
        self.value = sigmoid(self.value)

    def __repr__(self):
        return str(self.value)
        