from skynet_ml.metrics.metric import Metric
import numpy as np


class MAE(Metric):
    """
    MAE (Mean Absolute Error) metric class for evaluating regression tasks.

    The MAE class computes the mean absolute error between true (observed)
    values and the values predicted by a model. It provides a measure of
    the average magnitude of errors between predicted and observed values,
    without considering their direction. Each individual difference between 
    predicted and true values is weighted equally.

    Attributes
    ----------
    name : str
        Name of the metric ('mae').

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute the mean absolute error based on true and predicted values.

    Example
    -------
    >>> from skynet_ml.metrics import MAE
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 3.2])
    >>> mae = MAE()
    >>> error = mae.compute(y_true, y_pred)
    >>> print(error)
    0.13333333333333333
    """


    def __init__(self) -> None:
        """
        Initialize the MAE object with the name attribute set to 'mae'.
        """

        self.name = "mae"

        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the Mean Absolute Error (MAE) between true and predicted values.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated target values as returned by a regressor.

        Returns
        -------
        float
            The computed mean absolute error.

        Example
        -------
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([0.8, 2.1, 3.2])
        >>> mae = MAE()
        >>> error = mae.compute(y_true, y_pred)
        >>> print(error)
        0.13333333333333333
        """
        
        # Ensure that y_true and y_hat have the same shape
        self.check_shape(y_true, y_hat)

        # Calculate and return the mean absolute error
        return np.mean(np.abs(y_true - y_hat))
