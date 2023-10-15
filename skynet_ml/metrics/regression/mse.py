from skynet_ml.metrics.base import BaseMetric
import numpy as np


class MeanSquaredError(BaseMetric):
    """
    Mean Squared Error (MSE) metric.

    MSE is a measure of the average of the squares of the errors or deviations, 
    that is, the difference between the estimator and what is estimated.

    Attributes:
        name (str): Name of the metric.
    """


    def __init__(self) -> None:
        """
        Initializes the Mean Squared Error metric object.
        """
        self.name = "mean_squared_error"


    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the Mean Squared Error.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Returns:
            float: The Mean Squared Error value. Lower values indicate better 
                  model performance, with 0 representing a perfect prediction.

        Note:
            MSE provides a gross idea of the magnitude of error and is 
            always non-negative, with values closer to zero being better.
        """
        self.check_shape(y_true, y_pred)
        return np.mean(np.square(y_true - y_pred))
