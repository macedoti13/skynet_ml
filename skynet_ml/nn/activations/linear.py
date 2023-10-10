from skynet_ml.nn.activations.activation import Activation
import numpy as np


class Linear(Activation):
    """
    The Linear Activation function.

    The Linear function is an activation function that doesn’t modify the 
    input values. It is also known as identity activation function. It is
    mostly used in the output layer of a regression network.

    Attributes
    ----------
    name : str
        Name of the activation function.

    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the Linear activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the Linear activation with respect to its input.

    Example
    -------
    >>> linear = Linear()
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = linear.compute(input_array)
    >>> gradient_array = linear.gradient(input_array)
    """
    
    
    def __init__(self) -> None:
        """
        Initialize the Linear object with the name attribute set to 'Linear'.
        """
        self.name = "Linear"

        
    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the Linear activation function.

        This function will return an array with the same shape as the input, 
        essentially performing no operation on it.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The output of the Linear activation function, same as input `z`.
        """
        self._check_shape(z)
        return z
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the Linear activation with respect to its input.

        The gradient is one everywhere.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the Linear activation with respect to its input `z`.
        """
        self._check_shape(z)
        return np.ones_like(z)
