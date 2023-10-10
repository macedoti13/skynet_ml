from skynet_ml.metrics.metric import Metric
import numpy as np


class MSE(Metric):
    """
    MSE (Mean Squared Error) metric class for evaluating regression tasks.

    The MSE class provides a method to compute the mean squared error between 
    true (observed) values and the values predicted by a model. This metric is 
    widely used in regression analysis and provides an indication of the 
    accuracy of predictions in terms of the average squared deviation from 
    the true values.

    Attributes
    ----------
    name : str
        Name of the metric ('mse').

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute the mean squared error based on true and predicted values.

    Example
    -------
    >>> from skynet_ml.metrics import MSE
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 3.2])
    >>> mse = MSE()
    >>> error = mse.compute(y_true, y_pred)
    >>> print(error)
    0.026666666666666665
    """


    def __init__(self) -> None:
        """
        Initialize the MSE object with the name attribute set to 'mse'.
        """
        
        self.name = "mse"

        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the Mean Squared Error (MSE) for the provided ground truth and predicted values.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated target values as returned by a regressor.

        Returns
        -------
        float
            The computed mean squared error.

        Example
        -------
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([0.8, 2.1, 3.2])
        >>> mse = MSE()
        >>> error = mse.compute(y_true, y_pred)
        >>> print(error)
        0.026666666666666665
        """
        
        # Check if the shape of y_true and y_hat are identical
        self.check_shape(y_true, y_hat)

        # Calculate and return the mean squared error
        return np.mean((y_true - y_hat) ** 2)
