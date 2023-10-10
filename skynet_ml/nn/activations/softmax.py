from skynet_ml.nn.activations.activation import Activation
import numpy as np


class Softmax(Activation):
    """
    The Softmax Activation Function.

    Softmax is often used as the activation for the last layer of a classification 
    network because the result could be interpreted as a probability distribution.

    The Softmax function outputs a vector that represents the probability distributions 
    of a list of potential outcomes. It is a type of sigmoid function and is used 
    in various multiclass classification methods.

    Attributes
    ----------
    name : str
        Name of the activation function.

    Methods
    -------
    get_config() -> dict
        Retrieve the configuration of the activation function.
    compute(z: np.array) -> np.array
        Compute the forward pass of the Softmax activation function.
    gradient(z: np.array) -> np.array
        Compute the gradient of the Softmax activation with respect to its input.

    Example
    -------
    >>> softmax_activation = Softmax()
    >>> input_array = np.array([[2, -1], [-3, 4]])
    >>> output_array = softmax_activation.compute(input_array)
    >>> gradient_array = softmax_activation.gradient(input_array)
    """


    def __init__(self) -> None:
        """
        Initialize the Softmax object with the name attribute set to 'Softmax'.
        """
        self.name = "Softmax"
        

    def compute(self, z: np.array) -> np.array:
        """
        Compute the forward pass of the Softmax activation function.

        Softmax function converts the raw logits into probabilities by taking the 
        exponential of each class score and then normalizing it.

        Parameters
        ----------
        z : np.array
            The raw logits or input to the activation function.

        Returns
        -------
        np.array
            The probabilities after applying Softmax.
        """
        self._check_shape(z)
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting the max for numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        Compute the gradient of the Softmax activation with respect to its input.

        Note: When Softmax is used in conjunction with Cross Entropy loss during backpropagation,
        the derivative simplifies to (prediction - target). Therefore, it's recommended
        to use Cross Entropy loss for training the network when Softmax is used.

        Parameters
        ----------
        z : np.array
            The raw logits or input to the activation function.

        Returns
        -------
        np.array
            Ones-like array with the same shape as input `z`. The actual computation of the gradient 
            should be handled during the backpropagation process considering the loss function used.
        """
        self._check_shape(z)
        return np.ones_like(z)  # When combined with cross entropy, uses yhat - y simplification.
