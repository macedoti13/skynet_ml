from skynet_ml.nn.initialization_methods.initializer import Initializer
import numpy as np

class ConstantInitializer(Initializer):
    """
    Neural network weight initializer that sets all weights to a constant value.

    The `ConstantInitializer` class is used to initialize the weights of neural 
    network layers to a constant value. While not commonly used for deep networks 
    as it may hinder convergence during training, it can be useful in specific 
    scenarios or for testing purposes.

    The primary use of constant initialization is to check the behavior of the 
    optimization algorithm. For instance, it can be used to verify if the 
    algorithm can navigate from a poor starting point.

    Parameters
    ----------
    constant : float, optional
        The value to which all weights will be initialized. Default is 1.0.

    Attributes
    ----------
    constant : float
        The value to which all weights will be initialized.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Initialize the weights matrix for a layer with the constant value.
    """
    
    def __init__(self, constant: float = 1.0) -> None:
        """
        Initialize the ConstantInitializer with a given constant value.

        Parameters
        ----------
        constant : float, optional
            The value to which all weights will be initialized. Default is 1.0.
        """
        self.constant = constant
        self.name = "Constant"
        
        
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initialize the weights matrix of a neural network layer to a constant value.

        This method creates a matrix of the given shape (`input_dim`, `n_units`) and
        fills it with the predefined constant value. 

        Parameters
        ----------
        input_dim : int
            Number of input features/dimensions for the neural network layer.
        
        n_units : int
            Number of neurons/units in the current neural network layer.

        Returns
        -------
        np.array
            A matrix filled with the constant value, with shape (`input_dim`, `n_units`).
        """
        return np.full((input_dim, n_units), self.constant)
