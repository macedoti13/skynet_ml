from skynet_ml.metrics.classification.precision import Precision
from skynet_ml.metrics.classification.recall import Recall
from skynet_ml.metrics.metric import Metric
import numpy as np


class FScore(Metric):
    """
    FScore metric class for evaluating classification tasks.

    The FScore class calculates the F-Score for binary, multiclass, and 
    multilabel classification tasks. The F-Score is the weighted harmonic mean 
    of precision and recall, with a range of best value at 1 and worst at 0.

    Parameters
    ----------
    threshold : float, optional, default=0.5
        Decision threshold used for binary and multilabel tasks. Predictions 
        with probability greater than or equal to threshold are treated as 
        positive class.
    
    task_type : str, default='binary'
        Classification task type. Should be one of the following: 'binary', 
        'multiclass', or 'multilabel'.
    
    f : int, default=1
        Weight factor for recall in the F-Score calculation. When f is 1, 
        the F-Score is the harmonic mean of precision and recall.

    Attributes
    ----------
    name : str
        Name of the metric, initialized to 'fscore'.
    precision : Precision
        An instance of the Precision class.
    recall : Recall
        An instance of the Recall class.
    f : int
        Weight factor for recall in the F-Score calculation.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute the F-Score based on true and predicted labels.

    Example
    -------
    >>> from skynet_ml.metrics import FScore
    >>> y_true = np.array([1, 0, 1, 0, 1])
    >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
    >>> f_score = FScore(threshold=0.5, task_type='binary')
    >>> score = f_score.compute(y_true, y_pred)
    >>> print(score)
    0.8
    """
    
    
    def __init__(self, threshold: float = 0.5, task_type: str = "binary", f: int = 1) -> None:
        """
        Initialize the FScore object with specified threshold, task type, and F factor.
        """
        
        self.precision = Precision(threshold=threshold, task_type=task_type)
        self.recall = Recall(threshold=threshold, task_type=task_type)
        self.task_type = task_type
        self.name = "fscore"
        self.f = f
        
        
    def get_config(self) -> dict:
        """
        Return the configuration of the metric.

        Returns
        -------
        dict
            Configuration of the metric.
        """
        
        return {"task_type": self.task_type, "f": self.f}
        
        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the F-Score for the provided ground truth and predicted labels.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels.
        y_hat : np.array
            Estimated target labels as returned by a classifier.

        Returns
        -------
        float
            The computed F-Score value.

        Example
        -------
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
        >>> f_score = FScore(threshold=0.5, task_type='binary')
        >>> score = f_score.compute(y_true, y_pred)
        >>> print(score)
        0.8
        """
        
        precision = self.precision.compute(y_true=y_true, y_hat=y_hat)
        recall = self.recall.compute(y_true=y_true, y_hat=y_hat)
        
        denominator = ((self.f**2 * precision) + recall)
        return np.round((1 + self.f**2) * ((precision * recall)) / denominator, decimals=4) if denominator != 0 else 0
    