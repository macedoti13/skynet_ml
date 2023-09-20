from skynet_ml.nn.initialization_methods.initializer import Initializer
import numpy as np

class UniformInitializer(Initializer):
    """
    Initializer that generates weights from a uniform distribution.

    Neural network weights can be initialized in several ways to improve training. This
    initializer sets the weights by randomly sampling from a uniform distribution within 
    a specified range.

    The uniform initialization method can be beneficial, especially in networks with 
    activation functions such as the hyperbolic tangent.

    Attributes
    ----------
    low_limit : float
        The lower limit of the uniform distribution.
        
    high_limit : float
        The upper limit of the uniform distribution.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized from a uniform distribution.

    """
    
    def __init__(self, low_limit: float = -1.0, high_limit: float = 1.0) -> None:
        """
        Initialize the UniformInitializer with specified limits.

        Parameters
        ----------
        low_limit : float, optional
            The lower limit of the uniform distribution, by default -1.0.
        
        high_limit : float, optional
            The upper limit of the uniform distribution, by default 1.0.

        """
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.name = "Uniform"
        
        
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights from a uniform distribution.

        This method creates a 2D array of weights, initialized by drawing 
        random samples from a uniform distribution within the specified range 
        [low_limit, high_limit].

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
        return np.random.uniform(self.low_limit, self.high_limit, (input_dim, n_units))
