from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class ReLU(BaseActivation):
    """
    Rectified Linear Unit (ReLU) activation function. 
    Computes the function f(x) = max(0, x).
    
    Notes:
        - It acts as a pass-through for positive values and returns 0 for negative values.
        - Outperforms sigmoid and tanh activations in deep neural networks due to reduced impact from the vanishing gradient problem.
        - The gradient is 1 for x > 0, 0 for x < 0, and undefined for x = 0.
    """    
    
    
    def __init__(self) -> None:
        """
        Initializes the activation function.
        """        
        self.name = "relu"
        
        
    def compute(self, z: np.array) -> np.array:
        """
        Computes the ReLU activation for the given input array.

        Args:
            z (np.array): Input array to the activation function. 
                          Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The activated output, with negative values clipped to 0 and positive values unchanged.
        """        
        self._check_shape(z)
        return np.maximum(z, 0)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the ReLU activation for the given input array.

        Args:
            z (np.array): Input array to the activation function. 
                          Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The gradient, which is 1 for x > 0 and 0 for x < 0. Undefined for x = 0.
        """        
        self._check_shape(z)
        return np.where(z > 0, 1, 0)
