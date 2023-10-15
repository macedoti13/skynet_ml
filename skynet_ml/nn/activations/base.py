from abc import ABC, abstractmethod
import numpy as np


class BaseActivation(ABC):
    """
    Base class for activation functions. All custom activation functions 
    must inherit from this class.
    """
    
    @abstractmethod
    def compute(self, z: np.array) -> np.array:
        """
        Computes the activation for the given input array.

        Args:
            z (np.array): Input array to the activation function. 
                          Must be a 2D array of shape (batch_size, n_units).

        Returns:
            np.array: The output of the activation function for the given input array.
                      Has the same shape as the input array.
        """        
        pass

    
    @abstractmethod
    def gradient(self, z: np.array) -> np.array:
        """
        Computes the gradient of the activation function for the given input array.
        Represents the derivative of the activation function with respect to its input.

        Args:
            z (np.array): Input array to the activation function. 
                          Must be a 2D array of shape (batch_size, n_units).

        Returns:
            np.array: The gradient of the activation function for the given input array.
                      Has the same shape as the input array.
        """        
        pass


    @classmethod
    def _check_shape(cls, z: np.array) -> None:
        """
        Checks if the input array has the correct shape. 
        Raises a ValueError if the shape is incorrect.

        Args:
            z (np.array): Input array to the activation function. 
                          Must be a 2D array of shape (batch_size, n_units).

        Raises:
            ValueError: If the input array doesn't have the expected 2D shape.
        """        
        if z.ndim != 2:
            raise ValueError(f"Expected z to be a 2D array, but got {z.ndim}D array instead.")
