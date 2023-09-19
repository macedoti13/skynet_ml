from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    """
    Base class for neural network weight and bias initializers.

    Initializers define the way to set the initial random weights of 
    neural network layers. Proper initialization methods ensure effective 
    and efficient training of the neural model. These methods help in 
    faster convergence and in preventing issues like vanishing or 
    exploding gradients.

    Subclasses should implement the `initialize_weights` method to provide
    specific initialization strategies, such as Xavier, He, or random 
    initialization. By default, biases are initialized to zeros using the
    `initialize_bias` method, but it can be overridden in subclasses if a 
    different behavior is desired.

    Attributes
    ----------
    None

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Initialize the weights matrix for a layer based on the input dimensions 
        and the number of units.

    initialize_bias(n_units: int) -> np.array:
        Initialize the bias vector for a layer based on the number of units.
    """
    
    @abstractmethod
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initialize the weights for a given layer.

        Parameters
        ----------
        input_dim : int
            The number of input features or units from the previous layer.
        
        n_units : int
            The number of units in the current layer for which weights are being initialized.
        
        Returns
        -------
        np.array
            Initialized weights with shape (input_dim, n_units).
        """
        pass

    
    @classmethod
    def initialize_bias(cls, n_units: int) -> np.array:
        """
        Initialize the biases for a given layer.

        Biases are, by default, initialized to zeros. Override this method 
        in subclasses for different behaviors.

        Parameters
        ----------
        n_units : int
            The number of units in the layer for which biases are being initialized.
        
        Returns
        -------
        np.array
            Initialized biases with shape (1, n_units).
        """
        return np.zeros((1, n_units))
