from skynet_ml.nn.activation_functions.activation import Activation
import numpy as np

class ReLUActivation(Activation):
    """
    Implementation of the Rectified Linear Unit (ReLU) activation function for neural networks.

    The ReLU activation function is defined as:
        f(z) = max(0, z)

    The ReLU function outputs the input directly if it's positive, otherwise, it outputs zero.
    It's known for being computationally efficient and is widely used, especially in the context of 
    deep neural networks. However, it may suffer from the "dying ReLU" problem where neurons can 
    sometimes get stuck during training and cease to update.

    Attributes
    ----------
    None

    Methods
    -------
    compute(z: np.array) -> np.array:
        Compute the ReLU activation function for a given input.

    gradient(z: np.array) -> np.array:
        Compute the derivative of the ReLU activation function for a given input.

    """
    def __init__(self) -> None:
        """
        Initialize the ReLUActivation class.
        """
        self.name = "ReLU"


    def compute(self, z: np.array) -> np.array:
        """
        Compute the ReLU activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Output after applying the ReLU activation function.

        """
        self._check_shape(z)
        return np.maximum(z, 0)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the derivative of the ReLU activation function for a given input.

        Parameters
        ----------
        z : np.array
            Input data, typically the weighted sum of inputs and weights.

        Returns
        -------
        np.array
            Derivative of the ReLU activation function for each input value. Outputs 1 for positive values and 0 for non-positive values.

        """
        self._check_shape(z)
        return np.where(z > 0, 1, 0)
