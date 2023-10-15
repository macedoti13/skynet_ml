from abc import ABC, abstractmethod
import numpy as np


class BaseInitializer(ABC): 
    """
    Abstract base class for neural network weight and bias initializers.

    This class provides a foundation for defining custom initialization strategies. Derived classes 
    should implement the `initialize_weights` method to provide a specific initialization method.

    It's important to choose a proper initialization technique since it can significantly affect the 
    convergence speed and performance of a neural network.
    """    


    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Abstract method for initializing weights.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which the weights need to be initialized.

        Returns:
            np.array: Initialized weights matrix of shape (input_dim, n_units).
        """        
        pass
    
    
    @classmethod
    def initialize_bias(cls, n_units: int) -> np.array:
        """
        Initialize biases for the given number of units. By default, biases are initialized to zeros.

        Args:
            n_units (int): Number of units in the current layer for which biases need to be initialized.

        Returns:
            np.array: Biases vector of shape (1, n_units) initialized to zeros.
        """        
        return np.zeros((1, n_units))
