from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class TanhActivation(Activation):
    """
    Implementation of the hyperbolic tangent (tanh) activation function for neural networks.

    The tanh function is similar to the sigmoid but outputs values between -1 and 1. 
    It is defined as:
        tanh(z) = (2 / (1 + exp(-2z))) - 1

    It is often preferred over the sigmoid function for hidden layers because its 
    outputs are zero-centered. However, like the sigmoid, it can also suffer from 
    the vanishing gradient problem for very large or very small inputs.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the tanh activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the tanh activation function for a given input.

    """

    def compute(self, z: np.array) -> np.array:
        """
        Compute the tanh activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the tanh activation function.

        """
        self._check_shape(z)
        return np.tanh(z)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the tanh activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the tanh activation function for each input value.

        """
        self._check_shape(z)
        tanh = self.compute(z)
        return 1 - tanh ** 2
