from skynet_ml.nn.activations.activation import Activation
import numpy as np


class ReLU(Activation):
    """
    The Rectified Linear Unit (ReLU) Activation function.

    The ReLU function is a type of activation function that is widely used in
    convolutional neural networks and deep learning models. The function returns 0 
    if it receives any negative input, but for any positive value `x` it returns that 
    value back.

    Attributes
    ----------
    name : str
        Name of the activation function.

    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the ReLU activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the ReLU activation with respect to its input.

    Example
    -------
    >>> relu = ReLU()
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = relu.compute(input_array)
    >>> gradient_array = relu.gradient(input_array)
    """
    
    
    def __init__(self) -> None:
        """
        Initialize the ReLU object with the name attribute set to 'ReLU'.
        """
        self.name = "ReLU"
        
        
    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the ReLU activation function.

        This function will return an array with the same shape as the input, where each
        element is the input if it's positive, otherwise zero.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The output of the ReLU activation function.
        """
        self._check_shape(z)
        return np.maximum(z, 0)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the ReLU activation with respect to its input.

        The gradient is one for positive inputs and zero for non-positive inputs.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the ReLU activation with respect to its input `z`.
        """
        self._check_shape(z)
        return np.where(z > 0, 1, 0)
