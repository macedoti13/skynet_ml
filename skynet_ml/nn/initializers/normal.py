from skynet_ml.nn.initializers.initializer import Initializer
import numpy as np


class Normal(Initializer):
    """
    Normal (Gaussian) weight initializer.

    This class initializes weights with a normal distribution with specified mean and standard deviation.
    It is commonly used in neural networks, especially for initializing the weights of densely connected layers.

    Parameters
    ----------
    mean : float, optional
        The mean (center) of the normal distribution used for weight initialization. Default is 0.0.
    std : float, optional
        The standard deviation (spread) of the normal distribution used for weight initialization. Default is 1.0.

    Attributes
    ----------
    mean : float
        The mean of the normal distribution used for weight initialization.
    std : float
        The standard deviation of the normal distribution used for weight initialization.
    name : str
        The name identifier of the initializer, set to 'Normal'.

    Methods
    -------
    get_config() -> dict
        Returns a dictionary containing the mean and standard deviation of the normal distribution used for weight initialization.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights with a normal distribution of given mean and standard deviation.

    Examples
    --------
    To initialize weights with a normal distribution with mean 0 and standard deviation 0.05:

    >>> initializer = Normal(mean=0, std=0.05)
    >>> weights = initializer.initialize_weights(5, 10)
    """
    

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Initializes a `Normal` instance with specified mean and standard deviation for the normal distribution.

        Parameters
        ----------
        mean : float, optional
            The mean of the normal distribution used for weight initialization. Default is 0.0.
        std : float, optional
            The standard deviation of the normal distribution used for weight initialization. Default is 1.0.
        """
        self.mean = mean
        self.std = std
        self.name = "Normal"
        
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the normal initializer as a dictionary.

        The dictionary contains two key-value pairs, where the keys are 'mean' and 'std', and the values are
        the mean and standard deviation of the normal distribution used for weight initialization, respectively.

        Returns
        -------
        dict
            A dictionary containing the configuration of the normal initializer.
        """
        return {"mean": self.mean, "std": self.std}
        
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using a normal (Gaussian) distribution with specified mean and standard deviation.

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
        return np.random.normal(self.mean, self.std, (input_dim, n_units))
