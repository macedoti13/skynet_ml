from skynet_ml.metrics.base import BaseMetric
import numpy as np

class RootMeanSquaredError(BaseMetric):
    """
    Root Mean Squared Error (RMSE) metric.

    RMSE is a measure of the differences between predicted values by the model and 
    the actual values. It represents the square root of the second sample moment of the 
    differences between predicted values and observed values or the quadratic mean of 
    these differences.

    Attributes:
        name (str): Name of the metric.
    """


    def __init__(self) -> None:
        """
        Initializes the Root Mean Squared Error metric object.
        """
        self.name = "root_mean_squared_error"


    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the Root Mean Squared Error.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Returns:
            float: The Root Mean Squared Error value. Lower values indicate better 
                  model performance, with 0 representing a perfect prediction.

        Note:
            This metric gives more weight to larger errors than smaller ones, 
            penalizing large errors more.
        """
        self.check_shape(y_true, y_pred)
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
