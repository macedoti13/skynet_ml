from skynet_ml.nn.initializers.initializer import Initializer
import numpy as np


class XavierNormal(Initializer):
    """
    Xavier Normal weight initializer (also known as Glorot Normal initializer).

    This class initializes weights with a zero mean and a variance of 1/n, where n is the number of input units.
    It is effective for networks in which the activation function for hidden units is a sigmoid or hyperbolic tangent.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        The name identifier of the initializer, set to 'Xavier Normal'.

    Methods
    -------
    get_config() -> dict
        Returns an empty dictionary as there are no hyperparameters for this initializer.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights with zero mean and a variance of 1/n.

    Examples
    --------
    >>> initializer = XavierNormal()
    >>> weights = initializer.initialize_weights(5, 10)
    """


    def __init__(self) -> None:
        """
        Initializes a `XavierNormal` instance.
        """
        self.name = "Xavier Normal"
        
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using a normal distribution with zero mean and a variance of 1/n.

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
        std = np.sqrt(1.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class XavierUniform(Initializer):
    """
    Xavier Uniform weight initializer (also known as Glorot Uniform initializer).

    This class initializes weights with values drawn from a uniform distribution in the interval [-limit, limit],
    where limit is sqrt(6 / (n_in + n_out)) and n_in is the number of input units.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        The name identifier of the initializer, set to 'Xavier Uniform'.

    Methods
    -------
    get_config() -> dict
        Returns an empty dictionary as there are no hyperparameters for this initializer.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns the weights using Xavier Uniform initialization.

    Examples
    --------
    >>> initializer = XavierUniform()
    >>> weights = initializer.initialize_weights(5, 10)
    """


    def __init__(self) -> None:
        """
        Initializes a `XavierUniform` instance.
        """
        self.name = "Xavier Uniform"
        
        
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights using Xavier Uniform initialization.

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
        limit = np.sqrt(6.0 / (input_dim + n_units))
        return np.random.uniform(-limit, limit, (input_dim, n_units))
