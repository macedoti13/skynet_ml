from skynet_ml.metrics.metric import Metric
import numpy as np


class RSquared(Metric):
    """
    RSquared (Coefficient of Determination) class for evaluating regression tasks.

    The RSquared class computes the R^2 score, a statistic that provides a measure 
    of how well observed outcomes are replicated by the model. The best possible score is 1.0. 
    R^2 can be negative (because the model can be arbitrarily worse); a model that always predicts 
    the mean of y would get an R^2 score of 0.

    Attributes
    ----------
    name : str
        Name of the metric ('r2').

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute the R-squared based on true and predicted values.

    Example
    -------
    >>> from skynet_ml.metrics import RSquared
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 3.2])
    >>> r2 = RSquared()
    >>> score = r2.compute(y_true, y_pred)
    >>> print(score)
    0.94
    """


    def __init__(self) -> None:
        """
        Initialize the RSquared object with the name attribute set to 'r2'.
        """
        
        self.name = "r2"
    
    
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the Coefficient of Determination (R^2) between true and predicted values.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated target values as returned by a regressor.

        Returns
        -------
        float
            The computed R^2 score.

        Example
        -------
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([0.8, 2.1, 3.2])
        >>> r2 = RSquared()
        >>> score = r2.compute(y_true, y_pred)
        >>> print(score)
        0.94
        """
        
        # Ensure that y_true and y_hat have the same shape
        self.check_shape(y_hat, y_true)

        mean = np.mean(y_true, axis=0)  # mean of true values
        total_variance = np.sum((y_true - mean)**2, axis=0)  # total variance of true values
        residual_variance = np.sum((y_true - y_hat)**2, axis=0)  # variance not explained by model

        # Compute the R^2 scores per dimension, avoiding division by zero
        r2_scores = np.zeros_like(total_variance)
        non_zero_variance = total_variance != 0
        r2_scores[non_zero_variance] = 1 - (residual_variance[non_zero_variance] / total_variance[non_zero_variance])
        
        # Return the average R^2 score across dimensions
        return np.mean(r2_scores)
