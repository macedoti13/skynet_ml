from skynet_ml.metrics.metric import Metric
import numpy as np

class RSquaredMetric(Metric):
    """
    Metric class that calculates the R-squared (coefficient of determination) value between predicted and true labels.

    R-squared quantifies the proportion of the variance in the dependent variable that 
    is predictable from the independent variable(s). It provides a measure of how well 
    the model's predictions match the observed data.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the R-squared value for given predictions and true labels.
    """
    def __init__(self) -> None:
        self.name = "r2"

    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the R-squared value for given predictions and true labels. For multi-output 
        predictions, the variance is calculated for each dimension separately, and the 
        final R-squared value returned is the average across all dimensions.

        Parameters
        ----------
        yhat : np.array
            Predicted labels or outputs. Can be a single output (1-D array) or multi-output
            (2-D array where each column represents a different output).

        ytrue : np.array
            True labels. Should match the shape of yhat.

        Returns
        -------
        float
            The computed R-squared value. For single output predictions, it will be a 
            single value. For multi-output predictions, it will be the average R-squared
            value across all outputs.

        Notes
        -----
        The method first calculates the variance for each dimension (or output) 
        separately. For single output predictions, this is straightforward. For 
        multi-output, the mean and variance are calculated for each dimension 
        (or output). The final R-squared value for multi-output predictions is the 
        average across all outputs.

        """
        self._check_shape(yhat, ytrue)

        mean = np.mean(ytrue, axis=0)  # mean of true labels across each dimension
        total_variance = np.sum((ytrue - mean)**2, axis=0)  # total variance of true labels per dimension
        residual_variance = np.sum((ytrue - yhat)**2, axis=0)  # variance not explained by model per dimension

        r2_scores = np.zeros_like(total_variance)
        non_zero_variance = total_variance != 0
        r2_scores[non_zero_variance] = 1 - (residual_variance[non_zero_variance] / total_variance[non_zero_variance])
        
        return np.mean(r2_scores)  # average the R^2 scores across dimensions
