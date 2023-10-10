from skynet_ml.nn.activations.activation import Activation
import numpy as np


class Sigmoid(Activation):
    """
    The Sigmoid Activation function.
    
    The Sigmoid function is a type of activation function that is traditionally
    used in binary classification problems. It squashes its input to be between
    0 and 1, which is also the reason why it’s used in the output layer of a binary 
    classification neural network.
    
    Attributes
    ----------
    name : str
        Name of the activation function.
    
    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the Sigmoid activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the Sigmoid activation with respect to its input.

    Example
    -------
    >>> sigmoid = Sigmoid()
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = sigmoid.compute(input_array)
    >>> gradient_array = sigmoid.gradient(input_array)
    """
    
    
    def __init__(self) -> None:
        """
        Initialize the Sigmoid object with the name attribute set to 'Sigmoid'.
        """
        self.name = "Sigmoid"
    

    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the Sigmoid activation function.
        
        The Sigmoid function squashes the input `z` to be between 0 and 1.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The output of the Sigmoid activation function.
        """
        self._check_shape(z)
        return 1 / (1 + np.exp(-z))
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the Sigmoid activation with respect to its input.
        
        This method computes the derivative of the Sigmoid function, which is used 
        during the backpropagation step in training of neural networks.

        Parameters
        ----------
        z : np.array
            The input to the activation function.

        Returns
        -------
        np.array
            The gradient of the Sigmoid activation with respect to its input `z`.
        """
        self._check_shape(z)
        sigmoid = self.compute(z)
        return sigmoid * (1 - sigmoid)
