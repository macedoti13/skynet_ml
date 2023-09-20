from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class LinearActivation(Activation):
    """
    Implementation of the linear activation function for neural networks.

    The linear activation function is identity in nature, meaning it will return the input as it is.
    Its formula is simply:
        f(z) = z

    This function can be useful for tasks like regression where the output is not necessarily bound 
    to a specific range. However, for deep networks, it might not introduce any non-linearity, which 
    can limit the expressiveness of the network.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the linear activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the linear activation function for a given input.

    """
    def __init__(self) -> None:
        """
        Initialize the LinearActivation class.
        """
        self.name = "Linear"


    def compute(self, z: np.array) -> np.array:
        """
        Compute the linear activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the linear activation function (which is the same as the input).

        """
        self._check_shape(z)
        return z
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the linear activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the linear activation function for each input value, which is always 1.

        """
        self._check_shape(z)
        return np.ones_like(z)
