from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class Sigmoid(BaseActivation):
    """
    Sigmoid activation function.
    Computes the function f(x) = 1 / (1 + exp(-x)).
    
    Notes:
        - It squashes the input to the range [0, 1]. Larger values are closer to 1 and smaller values are closer to 0.
        - The output can be interpreted as a probability.
        - Suffers from the vanishing gradient problem when the input is large or small.
        - Useful for binary or multi-label classification problems.
        - The gradient varies with the input value but is always between 0 and 0.25.
        - Its output is not zero-centered, which can be problematic for gradient-based optimization methods.
    """    

    
    def __init__(self) -> None:
        """
        Initializes the activation function.
        """        
        self.name = "sigmoid"
    

    def compute(self, z: np.array) -> np.array:
        """
        Computes the sigmoid activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The activated output, with values squashed to the range [0, 1].
        """        
        self._check_shape(z)
        return 1 / (1 + np.exp(-z))
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the sigmoid activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The gradient, which is sigmoid(z) * (1 - sigmoid(z)).
        """        
        self._check_shape(z)
        sigmoid = self.compute(z)
        return sigmoid * (1 - sigmoid)
