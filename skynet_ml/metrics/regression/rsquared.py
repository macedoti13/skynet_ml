from skynet_ml.metrics.base import BaseMetric
import numpy as np

class RSquared(BaseMetric):
    """
    R-squared (Coefficient of Determination) metric.

    R-squared measures the proportion of the variance in the dependent variable 
    that is predictable from the independent variable(s). It provides a measure 
    of how well observed outcomes are replicated by the model.

    Attributes:
        name (str): Name of the metric.
    """


    def __init__(self) -> None:
        """
        Initializes the R-squared metric object.
        """
        self.name = "rsquared"


    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the R-squared metric.

        Args:
            y_true (np.array): Ground truth (correct) target values.
            y_pred (np.array): Estimated target values.

        Returns:
            float: R-squared value. A value of 1 indicates perfect prediction, 
                  whereas a value of 0 indicates that the model does not perform 
                  any better than simply taking the mean of the target variable.

        Note:
            The R-squared metric can take on negative values if the model is worse 
            than a simple average.
        """
        self.check_shape(y_pred, y_true)

        mean = np.mean(y_true, axis=0)  # mean of true values
        total_variance = np.sum((y_true - mean)**2, axis=0)  # total variance of true values
        residual_variance = np.sum((y_true - y_pred)**2, axis=0)  # variance not explained by model

        # Compute the R^2 scores per dimension, avoiding division by zero
        r2_scores = np.zeros_like(total_variance)
        non_zero_variance = total_variance != 0
        r2_scores[non_zero_variance] = 1 - (residual_variance[non_zero_variance] / total_variance[non_zero_variance])

        # Return the average R^2 score across dimensions
        return np.mean(r2_scores)
