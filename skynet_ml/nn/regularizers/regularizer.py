from abc import ABC, abstractmethod
import numpy as np

class Regularizer(ABC):
    """
    Base class for Regularizers.

    Regularization techniques are pivotal in training neural networks, particularly
    in preventing overfitting. Overfitting happens when a neural network model 
    performs exceptionally well on training data but fails to generalize on new, 
    unseen data. One common reason is the magnitude of the coefficients (weights) 
    becoming too large. Regularizers add a penalty to the loss function based on 
    the size or complexity of the weights, discouraging them from reaching large values.

    A regularizer will provide two main methods: 
    - `forward`: Computes the regularization loss.
    - `backward`: Computes the gradient of the regularization with respect to the weights.

    Parameters
    ----------
    lambda_val : float, optional
        Regularization strength parameter. Determines the amount of regularization 
        to apply, with higher values meaning more regularization. Default is 0.0001.

    Methods
    -------
    forward(weights: np.array) -> np.array:
        Compute the regularization term for the provided weights.

    backward(weights: np.array) -> np.array:
        Compute the gradient of the regularization term with respect to the weights.

    """

    def __init__(self, lambda_val: float = 0.0001):
        self.lambda_val = lambda_val
        
        
    @abstractmethod
    def forward(self, weights: np.array) -> float:
        """
        Compute the regularization term for the provided weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        float:
            Regularization term based on the provided weights.
        """
        pass
        
        
    @abstractmethod
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            The weights of a particular layer in the neural network.

        Returns
        -------
        np.array
            Gradient of the regularization term with respect to the provided weights.
        """
        pass
