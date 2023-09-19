from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class SigmoidActivation(Activation):
    """
    Implementation of the sigmoid activation function for neural networks.

    The sigmoid function, also known as the logistic function, is defined as:
        sigmoid(z) = 1 / (1 + exp(-z))

    It outputs a value between 0 and 1 and is often used for binary classification 
    tasks. However, it can suffer from the vanishing gradient problem when inputs 
    are too positive or too negative.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the sigmoid activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the sigmoid activation function for a given input.

    """

    def compute(self, z: np.array) -> np.array:
        """
        Compute the sigmoid activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the sigmoid activation function.

        """
        self._check_shape(z)
        return 1 / (1 + np.exp(-z))
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the sigmoid activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the sigmoid activation function for each input value.

        """
        self._check_shape(z)
        sigmoid = self.compute(z)
        return sigmoid * (1 - sigmoid)
