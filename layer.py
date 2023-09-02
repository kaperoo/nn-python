# Layer class for neural network
import numpy as np
from node import Node

class Layer:
    def __init__(self, size, n_parents):
        self.size = size
        self.n_parents = n_parents
        self.nodes = np.array([])
        for _ in range(size):
            self.nodes = np.append(self.nodes, Node(np.random.normal(0.0, 1.0, n_parents), np.random.normal(0.0, 1.0)))

    def __repr__(self):
        return str(self.nodes)