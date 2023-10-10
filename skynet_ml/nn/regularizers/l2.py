from skynet_ml.nn.regularizers.regularizer import Regularizer
import numpy as np


class L2(Regularizer):
    """
    L2 Regularizer Class
    
    The L2 Regularizer class implements L2 (Ridge) regularization. L2 regularization
    discourages large weights in the model by adding the L2 norm (the square root of the 
    sum of the squared weights) multiplied by the regularization factor lambda to the loss function.
    
    L2 regularization can help prevent overfitting by discouraging overly complex models 
    that fit the training data too closely.

    Parameters
    ----------
    lambda_val : float, optional
        The regularization factor, default is set in the base `Regularizer` class.

    Methods
    -------
    forward(weights: np.array) -> float:
        Compute the L2 regularization term for the given weights.
    
    backward(weights: np.array) -> np.array:
        Compute and return the gradient of the L2 regularization term with respect to the weights.
    
    get_config() -> dict:
        Returns a dictionary containing the configuration of the regularizer.

    Example
    -------
    >>> l2_reg = L2(lambda_val=0.01)
    >>> weights = np.array([1.0, -2.0, 3.0])
    >>> print(l2_reg.forward(weights))  # Outputs: 0.14
    >>> print(l2_reg.backward(weights)) # Outputs: [ 0.02 -0.04  0.06]
    """


    def forward(self, weights: np.array) -> np.array:
        """
        Compute the L2 regularization term for the given weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the L2 regularization term is computed.

        Returns
        -------
        float
            The L2 regularization term for the given weights.
        """
        return self.lambda_val * np.sum(np.square(weights))
    
    
    def backward(self, weights: np.array) -> np.array:
        """
        Compute the gradient of the L2 regularization term with respect to the weights.

        Parameters
        ----------
        weights : np.array
            Weights of the model for which the gradient is computed.

        Returns
        -------
        np.array
            Gradient of the L2 regularization term with respect to the given weights.
        """
        return self.lambda_val * 2 * weights


    def get_config(self) -> dict:
        """
        Returns the configuration of the L2 regularizer.

        Returns
        -------
        dict
            A dictionary containing the regularization factor lambda.

        Example
        -------
        >>> l2_reg = L2(lambda_val=0.01)
        >>> config = l2_reg.get_config()
        >>> print(config)
        {'lambda_val': 0.01}
        """
        return {"lambda_val": self.lambda_val}
