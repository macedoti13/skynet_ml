from skynet_ml.metrics.base import BaseMetric
import numpy as np

class MeanAbsoluteError(BaseMetric):
    """
    Mean Absolute Error (MAE) metric.

    MAE measures the average of the absolute differences between predictions 
    and actual observations. It gives an idea of the magnitude of error, but 
    does not provide any information on the direction (over or under prediction).

    Attributes:
        name (str): Name of the metric.
    """

    def __init__(self) -> None:
        """
        Initializes the Mean Absolute Error metric object.
        """
        self.name = "mean_absolute_error"


    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the Mean Absolute Error.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Returns:
            float: The Mean Absolute Error value. Lower values indicate better 
                  model performance, with 0 representing a perfect prediction.

        Note:
            Unlike the Mean Squared Error, MAE is linear and will treat all 
            individual differences in the same way, making it less sensitive 
            to outliers than the MSE.
        """
        self.check_shape(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))
