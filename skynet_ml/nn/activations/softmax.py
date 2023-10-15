from skynet_ml.nn.activations.base import BaseActivation
import numpy as np


class Softmax(BaseActivation):
    """
    Computes the softmax function, f(x) = exp(x) / sum(exp(x)).
    
    Notes:
        - It squashes the input to the range [0, 1]. Larger values are closer to 1 and smaller values are closer to 0.
        - Acts as a multi-class version of the sigmoid function.
        - Functions as a probability distribution over the classes.
            - Acts in the entire output layer of a neural network. It gives the probability of each class that sums to 1.
    """    

    
    def __init__(self) -> None:
        """
        Initializes the activation function.
        """        
        self.name = "softmax"
        

    def compute(self, z: np.array) -> np.array:
        """
        Computes the softmax activation for the given input array.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: The activated output, with values normalized to represent probabilities in the range [0, 1].
        """        
        self._check_shape(z)
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting the max for numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def gradient(self, z: np.array) -> np.array:
        """
        This method doesn't provide the true gradient of the softmax function. 
        The actual gradient would be a Jacobian matrix of shape (n_units, n_units).
        In the context of neural networks, when combined with the cross entropy loss, 
        the gradient simplifies to (yhat - y). Hence, for simplicity and specific applications,
        this function returns a vector of ones, which when multiplied by (yhat - y) gives the desired gradient. 
        Note: This might not be applicable for all use cases.

        Args:
            z (np.array): The input array to the activation function. Expected to have a shape (batch_size, n_units).

        Returns:
            np.array: A vector of ones with the same shape as the input.
        """        
        self._check_shape(z)
        return np.ones_like(z)  # Used in combination with cross entropy's (yhat - y) simplification.
