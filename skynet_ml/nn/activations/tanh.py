from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class Tanh(BaseActivation):
    """
    Computes the hyperbolic tangent function, f(x) = tanh(x).
    
    Notes:
        - It squashes the input to the range [-1, 1]. Larger values are closer to 1 and smaller values are closer to -1.
        - Useful in hidden layers of neural networks.
        - Can suffer from the vanishing gradient problem, especially when inputs are far from the origin along the x-axis.
        - The gradient varies with the input but is always between 0 and 1, becoming very small for large absolute values of the input.
    """    

    
    def __init__(self) -> None:
        """
        Initializes the activation function.
        """        
        self.name = "tanh"


    def compute(self, z: np.array) -> np.array:
        """
        Computes the tanh activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The activated output, with values in the range [-1, 1].
        """        
        self._check_shape(z)
        return np.tanh(z)


    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the tanh activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The gradient, diminishing as the absolute value of the input grows, calculated as 1 - tanh(z) ** 2.
        """        
        self._check_shape(z)
        tanh = self.compute(z)
        return 1 - tanh ** 2
