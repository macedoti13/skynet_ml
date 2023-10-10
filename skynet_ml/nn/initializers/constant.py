from skynet_ml.nn.initializers.initializer import Initializer
import numpy as np


class Constant(Initializer):
    """
    Constant weight initializer.

    This initializer generates a tensor with all elements set to a constant value. It is primarily used for initializing 
    biases and can also be used for testing purposes to initialize weights with a constant value.

    Parameters
    ----------
    constant : float, optional
        The constant value to set the elements of initialized tensor to, default is 1.0.

    Attributes
    ----------
    constant : float
        The constant value used for initialization.
    name : str
        The name identifier of the initializer, set to 'Constant'.

    Methods
    -------
    get_config() -> dict
        Returns a dictionary containing the `constant` value.
    initialize_weights(input_dim: int, n_units: int) -> np.array
        Initializes and returns a tensor with all elements set to `constant`.

    Examples
    --------
    >>> initializer = Constant(constant=0.5)
    >>> weights = initializer.initialize_weights(5, 10)
    """
    
    
    def __init__(self, constant: float = 1.0) -> None:
        """
        Initializes the `Constant` initializer with the specified constant value.

        Parameters
        ----------
        constant : float, optional
            The constant value to use for initialization. Default is 1.0.
        """
        self.constant = constant
        self.name = "Constant"
        
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the `Constant` initializer as a dictionary.

        The dictionary contains a single key-value pair, mapping 'constant' to the value specified during initialization.

        Returns
        -------
        dict
            A dictionary containing the constant value used for initialization.
        """
        return {"constant": self.constant}
        
        
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes weights with a constant value.

        This method creates and returns a 2D numpy array with all elements set to the `constant` value specified during 
        initialization. The size of the initialized array is determined by `input_dim` and `n_units` parameters.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input data.
        n_units : int
            The number of units in the layer for which the weights are initialized.

        Returns
        -------
        np.array
            A 2D numpy array containing the initialized weights with a shape of (input_dim, n_units), with all elements
            set to the specified constant value.
        """
        return np.full((input_dim, n_units), self.constant)
