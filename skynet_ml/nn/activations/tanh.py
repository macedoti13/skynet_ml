from skynet_ml.nn.activations.activation import Activation
import numpy as np


class Tanh(Activation):
    """
    The Hyperbolic Tangent (Tanh) Activation Function.

    The Tanh function is used as an activation function in neural networks 
    due to its properties of outputting values between -1 and 1, which can be 
    useful in various scenarios, such as when the model needs to predict values 
    that are ordered (rankings), centered around 0.

    Attributes
    ----------
    name : str
        Name of the activation function.

    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the Tanh activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the Tanh activation with respect to its input.

    Example
    -------
    >>> tanh_activation = Tanh()
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = tanh_activation.compute(input_array)
    >>> gradient_array = tanh_activation.gradient(input_array)
    """
    
    
    def __init__(self) -> None:
        """
        Initialize the Tanh object with the name attribute set to 'Tanh'.
        """
        self.name = "Tanh"


    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the Tanh activation function.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The output of the Tanh activation function.
        """
        self._check_shape(z)
        return np.tanh(z)


    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the Tanh activation with respect to its input.

        The gradient is calculated as 1 - tanh^2(z).

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the Tanh activation with respect to its input `z`.
        """
        self._check_shape(z)
        tanh = self.compute(z)
        return 1 - tanh ** 2
