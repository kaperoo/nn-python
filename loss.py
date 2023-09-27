# Loss functions for training
import numpy as np


class MSE:
    def calculateError(self, y_hat, y):
        return np.square(np.subtract(y_hat, y)) / 2

    def derivativeError(self, y_hat, y):
        return np.subtract(y, y_hat)
