from skynet_ml.metrics.metric import Metric
import numpy as np


class RMSE(Metric):
    """
    RMSE (Root Mean Square Error) class for evaluating regression tasks.

    The RMSE class computes the RMSE score, a standard way to measure the error of a 
    model in predicting quantitative data. It is the square root of the average of 
    squared differences between prediction and actual observation. Lower values of 
    RMSE indicate better fit, but be aware that it is sensitive to outliers.

    Attributes
    ----------
    name : str
        Name of the metric ('rmse').

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute the RMSE based on true and predicted values.

    Example
    -------
    >>> from skynet_ml.metrics import RMSE
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 3.2])
    >>> rmse = RMSE()
    >>> score = rmse.compute(y_true, y_pred)
    >>> print(score)
    0.18257418583505536
    """


    def __init__(self) -> None:
        """
        Initialize the RMSE object with the name attribute set to 'rmse'.
        """
        
        self.name = "rmse"
        
        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated target values as returned by a regressor.

        Returns
        -------
        float
            The computed RMSE value.

        Example
        -------
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([0.8, 2.1, 3.2])
        >>> rmse = RMSE()
        >>> score = rmse.compute(y_true, y_pred)
        >>> print(score)
        0.1414
        """
        
        # Check if the shape of y_true and y_hat are identical
        self.check_shape(y_true, y_hat)

        # Calculate and return the mean squared error
        return np.sqrt(np.mean((y_true - y_hat) ** 2))
