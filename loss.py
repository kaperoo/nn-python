# Loss functions for training
import numpy as np

def mse(y, y_hat):
    return np.square(np.subtract(y, y_hat))