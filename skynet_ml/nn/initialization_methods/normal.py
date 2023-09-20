from skynet_ml.nn.initialization_methods.initializer import Initializer
import numpy as np

class NormalInitializer(Initializer):
    """
    Initializer that generates weights from a normal (Gaussian) distribution.

    Initializing neural network weights from a Gaussian distribution can be beneficial
    for certain architectures and activation functions. This initializer sets the weights
    by randomly sampling from a normal distribution with a specified mean and standard deviation.

    Attributes
    ----------
    mean : float
        The mean of the normal distribution.
        
    std : float
        The standard deviation of the normal distribution.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized from a normal distribution.

    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Initialize the NormalInitializer with specified mean and standard deviation.

        Parameters
        ----------
        mean : float, optional
            The mean of the normal distribution, by default 0.0.
        
        std : float, optional
            The standard deviation of the normal distribution, by default 1.0.

        """
        self.mean = mean
        self.std = std
        self.name = "Normal"
        
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights from a normal distribution.

        This method creates a 2D array of weights, initialized by drawing 
        random samples from a normal distribution with the specified mean and 
        standard deviation.

        Parameters
        ----------
        input_dim : int
            The number of input features or neurons.
        
        n_units : int
            The number of units or neurons in the layer.

        Returns
        -------
        np.array
            A 2D numpy array of initialized weights with shape (input_dim, n_units).

        """
        return np.random.normal(self.mean, self.std, (input_dim, n_units))
