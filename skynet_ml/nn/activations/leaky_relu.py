from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class LeakyReLU(BaseActivation):
    """
    Leaky Rectified Linear Unit (LeakyReLU) activation function.
    Computes the function f(x) = max(alpha * x, x) where alpha is a small constant, typically close to 0.
    
    Notes:
        - It acts as a pass-through for positive values and multiplies negative values by the constant alpha.
        - Outperforms sigmoid and tanh activations in deep neural networks due to reduced impact from the vanishing gradient problem.
        - Provides an advantage over ReLU by preventing dead neurons (neurons that never activate due to a negative gradient).
        - The gradient is 1 for x > 0, alpha for x < 0. While theoretically undefined for x = 0, in practice, it is treated as 1.
    """    
    
    
    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initializes the activation function.

        Args:
            alpha (float, optional): Small constant for negative input values. Typically close to 0. Defaults to 0.01.
        """        
        self.alpha = alpha
        self.name = "leaky_relu"

        
    def compute(self, z: np.array) -> np.array:
        """
        Computes the LeakyReLU activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The activated output, with negative values multiplied by alpha and positive values unchanged.
        """        
        self._check_shape(z)
        return np.where(z > 0, z, self.alpha * z)

    
    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the LeakyReLU activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The gradient, which is 1 for x > 0 and alpha for x < 0. Treated as 1 for x = 0 for computational purposes.
        """        
        self._check_shape(z)
        return np.where(z > 0, 1, self.alpha)
