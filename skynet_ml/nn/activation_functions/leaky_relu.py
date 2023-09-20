from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class LeakyReLUActivation(Activation):
    """
    Implementation of the Leaky Rectified Linear Unit (Leaky ReLU) activation function for neural networks.

    Leaky ReLU is an attempt to solve the dying ReLU problem. Instead of the function being zero 
    when z < 0, a leaky ReLU allows a small, non-zero, constant gradient alpha.

    The Leaky ReLU function is defined as:
        f(z) = z if z > 0, alpha * z otherwise.

    This modification ensures that all neurons in the network can receive updates, thus mitigating 
    the dying ReLU problem.

    Attributes
    ----------
    alpha : float
        The slope coefficient for negative values. Default value is 0.01.

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the Leaky ReLU activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the Leaky ReLU activation function for a given input.

    """

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initialize the LeakyReLUActivation class with a given alpha.

        Parameters
        ----------
        alpha : float, optional
            The slope coefficient for negative values. Default value is 0.01.
        """
        self.alpha = alpha
        self.name == "LeakyReLU"


    def compute(self, z: np.array) -> np.array:
        """
        Compute the Leaky ReLU activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the Leaky ReLU activation function.

        """
        self._check_shape(z)
        return np.where(z > 0, z, self.alpha * z)


    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the Leaky ReLU activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the Leaky ReLU activation function for each input value.

        """
        self._check_shape(z)
        return np.where(z > 0, 1, self.alpha)
