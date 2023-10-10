from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC):
    """
    Abstract Base Class for Regularization techniques.

    The Regularizer class is an abstract class that defines the basic interface
    that all regularizers must implement. Regularizers are techniques used to impose
    constraints on the weights during training, helping to mitigate overfitting.

    Parameters
    ----------
    lambda_val : float, optional
        Regularization factor. Higher values of lambda_val impose more regularization.
        Default value is 0.0001.

    Methods
    -------
    forward(weights: np.array) -> float:
        Compute the regularization penalty for the given weights.
    
    backward(weights: np.array) -> np.array:
        Compute the gradient of the regularization penalty with respect to the weights.
    
    get_config() -> dict:
        Returns a dictionary containing the configuration of the regularizer.

    Notes
    -----
    - Subclasses should implement the `forward`, `backward` and `get_config` methods.

    Example
    -------
    >>> class L2Regularizer(Regularizer):
    ...     def forward(self, weights):
    ...         return self.lambda_val * np.sum(weights ** 2)
    ...     def backward(self, weights):
    ...         return 2 * self.lambda_val * weights
    ...     def get_config(self):
    ...         return {"lambda_val": self.lambda_val}
    """
    

    def __init__(self, lambda_val: float = 0.0001):
        """
        Initializes the Regularizer with a specified lambda value.

        Parameters
        ----------
        lambda_val : float, optional
            The regularization factor, default is 0.0001.
        """
        self.lambda_val = lambda_val
        
        
    @abstractmethod
    def forward(self, weights: np.array) -> float:
        """
        Compute the regularization penalty for the given weights.

        Parameters
        ----------
        weights : np.array
            The weights for which the regularization penalty is to be computed.

        Returns
        -------
        float
            The computed regularization penalty.
        """
        pass
        
        
    @abstractmethod
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the regularization penalty with respect to the weights.

        Parameters
        ----------
        weights : np.array
            The weights for which the gradient of the regularization penalty is to be computed.

        Returns
        -------
        np.array
            The gradient of the regularization penalty with respect to the weights.
        """
        pass


    @classmethod
    def get_config(self) -> dict:
        """
        Returns a dictionary containing the configuration of the regularizer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the regularizer.

        Example
        -------
        >>> reg = L2Regularizer(lambda_val=0.01)
        >>> config = reg.get_config()
        >>> print(config)
        {'lambda_val': 0.01}
        """
        return {}
