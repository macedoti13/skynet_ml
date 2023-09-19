from skynet_ml.metrics.metric import Metric
import numpy as np

class MseMetric(Metric):
    """
    Metric that calculates the mean squared error (MSE) between predicted and true labels.
    
    The MSE quantifies the average of the squares of the differences between predicted and 
    true values. It is commonly used in regression problems to gauge the quality of a model's predictions.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the MSE value for given predictions and true labels.
    """
    
    def __init__(self) -> None:
        self.name = "mse"
    
    
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the MSE value for given predictions and true labels.

        For single output predictions, the MSE is calculated as the average of 
        the squared differences between true and predicted values. For multi-output 
        predictions, the MSE is computed for each output separately, and the final 
        value is the mean of these MSE values across all outputs.

        Batches: If the input arrays represent multiple samples (batches), the 
        computation takes into account all the samples to produce a single MSE value.

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
            The computed MSE value. 

        Raises
        ------
        TypeError:
            If the computed MSE is not of float type.
        """
        self._check_shape(yhat, ytrue)
        return np.mean((ytrue - yhat)**2)
