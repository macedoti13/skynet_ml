from skynet_ml.nn.initializers.initializer import Initializer
import numpy as np


class HeNormal(Initializer):
    """
    He Normal weight initializer.

    Also known as He et al. initializer, this initialization method is designed to work with ReLU (rectified linear unit) 
    activations by initializing weights with a standard deviation of `sqrt(2 / n)`, where `n` is the number of input 
    units in the weight tensor.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        The name identifier of the initializer, set to 'He Normal'.

    Methods
    -------
    get_config() -> dict
        Returns an empty dictionary as there are no hyperparameters for this initializer.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights with zero mean and a variance of 2/n.

    Examples
    --------
    >>> initializer = HeNormal()
    >>> weights = initializer.initialize_weights(5, 10)
    """
    
    
    def __init__(self) -> None:
        self.name = "He Normal"
    

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using the He Normal initialization.

        The weights are sampled from a normal distribution with mean zero and standard deviation `sqrt(2 / input_dim)`.
        The size of the initialized weight matrix is determined by `input_dim` and `n_units` parameters.

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
        std = np.sqrt(2.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class HeUniform(Initializer):
    """
    He Uniform weight initializer.

    Designed to keep the variance of the weights in each layer in a deep network with ReLU activations roughly the same.
    It initializes the weights with values drawn from a uniform distribution within [-limit, limit], where limit is 
    `sqrt(6 / n)` and `n` is the number of input units in the weight tensor.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        The name identifier of the initializer, set to 'He Uniform'.

    Methods
    -------
    get_config() -> dict
        Returns an empty dictionary as there are no hyperparameters for this initializer.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights using He Uniform initialization.

    Examples
    --------
    >>> initializer = HeUniform()
    >>> weights = initializer.initialize_weights(5, 10)
    """
    
    
    def __init__(self) -> None:
        """
        Initializes a `HeUniform` instance.
        """
        self.name = "He Uniform"
        

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using He Uniform initialization.

        The weights are sampled from a uniform distribution within the interval `[-limit, limit]` where the limit is 
        `sqrt(6 / input_dim)`. The size of the initialized weight matrix is determined by `input_dim` and `n_units` 
        parameters.

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
        limit = np.sqrt(6.0 / input_dim)
        return np.random.uniform(-limit, limit, (input_dim, n_units))
