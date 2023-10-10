from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np


class L1(Regularizer):
    """
    L1 Regularizer Class
    
    This class implements L1 regularization, which helps to encourage sparsity
    in the model parameters (weights). It's particularly useful for feature selection.
    
    L1 regularization adds the L1 norm of weights (the sum of the absolute values of
    the weights) multiplied by the regularization factor lambda to the loss function.

    Parameters
    ----------
    lambda_val : float, optional
        The regularization factor, default is set in the base `Regularizer` class.

    Methods
    -------
    forward(weights: np.array) -> float:
        Compute the L1 regularization term for the given weights.
    
    backward(weights: np.array) -> np.array:
        Compute and return the gradient of the L1 regularization term with respect to the weights.
    
    get_config() -> dict:
        Returns a dictionary containing the configuration of the regularizer.

    Example
    -------
    >>> l1_reg = L1(lambda_val=0.01)
    >>> weights = np.array([1.0, -2.0, 3.0])
    >>> print(l1_reg.forward(weights))  # Outputs: 0.06
    >>> print(l1_reg.backward(weights)) # Outputs: [ 0.01 -0.01  0.01]
    """


    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L1 regularization term for the given weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the L1 regularization term is computed.

        Returns
        -------
        float
            The L1 regularization term for the given weights.
        """
        return self.lambda_val * np.sum(np.abs(weights))
    
    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L1 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the gradient is computed.

        Returns
        -------
        np.array
            Gradient of the L1 regularization term with respect to the given weights.
        """
        return self.lambda_val * np.sign(weights)


    def get_config(self) -> dict:
        """
        Returns the configuration of the L1 regularizer.

        Returns
        -------
        dict
            A dictionary containing the regularization factor lambda.

        Example
        -------
        >>> l1_reg = L1(lambda_val=0.01)
        >>> config = l1_reg.get_config()
        >>> print(config)
        {'lambda_val': 0.01}
        """
        return {"lambda_val": self.lambda_val}
