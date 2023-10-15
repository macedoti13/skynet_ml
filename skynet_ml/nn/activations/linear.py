from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class Linear(BaseActivation):
    """
    Linear activation function that calculates the identity function, i.e., f(x) = x.

    This can be beneficial as the output activation function for regression problems or
    as the output activation function for classification problems when from_logits=True 
    for the loss function.

    Notes:
        - It's essentially a pass-through and does not transform the input in any way.
        - The gradient is always 1.
    """
    
    
    def __init__(self) -> None:
        """
        Initializes the activation function.
        """        
        self.name = "linear"

        
    def compute(self, z: np.array) -> np.array:
        """
        Computes the activation for the given input array.

        Args:
            z (np.array): Input array to the activation function. 
                          Should have a shape (batch_size, n_units).

        Returns:
            np.array: The output, which is identical to the input.
        """        
        # Ensures the input has the correct shape
        self._check_shape(z)
        return z
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the activation function for the given input array.

        Args:
            z (np.array): Input array to the activation function. 
                          Should have a shape (batch_size, n_units).

        Returns:
            np.array: The gradient, which is an array of ones with the same shape as the input.
        """        
        # Ensures the input has the correct shape
        self._check_shape(z)
        return np.ones_like(z)
