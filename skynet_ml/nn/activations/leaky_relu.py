from skynet_ml.nn.activations.activation import Activation
import numpy as np


class LeakyReLU(Activation):
    """
    The Leaky Rectified Linear Unit (ReLU) Activation function.

    The LeakyReLU is a variant of the ReLU function that allows small, 
    non-zero gradients when the unit is not active, which can help 
    mitigate the vanishing gradient problem in deep networks. The 
    'leakiness' of the activation function is controlled by the alpha 
    parameter, a small positive value.

    Attributes
    ----------
    alpha : float
        The slope coefficient for the negative part of the function. 
        Default is 0.01.
    name : str
        Name of the activation function.

    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the LeakyReLU activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the LeakyReLU activation with respect to its input.

    Example
    -------
    >>> leaky_relu = LeakyReLu(alpha=0.01)
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = leaky_relu.compute(input_array)
    >>> gradient_array = leaky_relu.gradient(input_array)
    """
    
    
    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initialize the LeakyReLU object with the slope coefficient alpha 
        and the name attribute set to 'LeakyReLU'.

        Parameters
        ----------
        alpha : float, optional
            The slope coefficient for the negative part of the function. 
            Default is 0.01.
        """
        self.alpha = alpha
        self.name = "LeakyReLU"
        
        
    def get_config(self):
        """
        Get the configuration of the ReLU activation function.

        Since ReLU activation function does not have hyperparameters, 
        this method returns an empty dictionary.

        Returns
        -------
        dict
            An empty dictionary.
        """
        return {"alpha": self.alpha}


    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the LeakyReLU activation function.

        For input values greater than zero, it returns the input value itself.
        For input values less than or equal to zero, it returns alpha times 
        the input value.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The output of the LeakyReLU activation function.
        """
        self._check_shape(z)
        return np.where(z > 0, z, self.alpha * z)


    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the LeakyReLU activation with respect to its input.

        The gradient is one for positive input values and alpha for negative input values.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the LeakyReLU activation with respect to its input `z`.
        """
        self._check_shape(z)
        return np.where(z > 0, 1, self.alpha)
