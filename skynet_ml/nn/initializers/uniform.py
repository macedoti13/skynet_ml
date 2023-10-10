from skynet_ml.nn.initializers.initializer import Initializer
import numpy as np


class Uniform(Initializer):
    """
    Uniform weight initializer.

    This class initializes weights with a uniform distribution between specified low and high limits.
    It is commonly used in neural networks, especially for initializing weights of layers.

    Parameters
    ----------
    low_limit : float, optional
        The lower bound on the range of random values generated to initialize the weights. Default is -1.0.
    high_limit : float, optional
        The upper bound on the range of random values generated to initialize the weights. Default is 1.0.

    Attributes
    ----------
    low_limit : float
        Lower limit of the uniform distribution used for weight initialization.
    high_limit : float
        Upper limit of the uniform distribution used for weight initialization.
    name : str
        The name identifier of the initializer, set to 'Uniform'.

    Methods
    -------
    get_config() -> dict
        Returns a dictionary containing the configuration of the uniform initializer (low and high limits).
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights with a uniform distribution between the low and high limits.

    Examples
    --------
    To initialize weights with a uniform distribution between -0.05 and 0.05:

    >>> initializer = Uniform(low_limit=-0.05, high_limit=0.05)
    >>> weights = initializer.initialize_weights(5, 10)
    """


    def __init__(self, low_limit: float = -1.0, high_limit: float = 1.0) -> None:
        """
        Initializes a `Uniform` instance with specified low and high limits for the uniform distribution.

        Parameters
        ----------
        low_limit : float, optional
            Lower limit of the uniform distribution used for weight initialization. Default is -1.0.
        high_limit : float, optional
            Upper limit of the uniform distribution used for weight initialization. Default is 1.0.
        """
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.name = "Uniform"
        
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the uniform initializer as a dictionary.

        The dictionary contains two key-value pairs, where the keys are 'low_limit' and 'high_limit', and the values are
        the lower and upper limits of the uniform distribution used for weight initialization, respectively.

        Returns
        -------
        dict
            A dictionary containing the configuration of the uniform initializer.
        """
        return {"low_limit": self.low_limit, "high_limit": self.high_limit}
        
        
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using a uniform distribution between specified low and high limits.

        The size of the initialized weight matrix is determined by the `input_dim` and `n_units` parameters.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input data.
        n_units : int
            The number of units in the layer for which the weights are initialized.

        Returns
        -------
        np.array
            A 2D numpy array containing the initialized weights with a shape of (input_dim, n_units).
        """
        return np.random.uniform(self.low_limit, self.high_limit, (input_dim, n_units))
