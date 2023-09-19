from skynet_ml.metrics.metric import Metric
import numpy as np

class AccuracyMetric(Metric):
    """
    Accuracy metric for classification tasks.

    Calculates the accuracy for binary, multilabel, or multiclass classification tasks based on
    the provided task type.

    Attributes
    ----------
    threshold : float
        Threshold value for binary and multilabel classification. Values above this threshold
        are considered as class 1, and values below or equal are considered as class 0.

    task_type : str
        The type of classification task. Must be one of 'binary', 'multiclass', or 'multilabel'.

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the accuracy based on the task type for the given predictions and true labels.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the accuracy metric with the specified threshold and task type.

        Parameters
        ----------
        threshold : float, optional
            The threshold for binary and multilabel classification. Default is 0.5.

        task_type : str, optional
            The type of classification task. Default is 'binary'.
        """
        self.threshold = threshold
        self.name = "accuracy"

        
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the accuracy based on the task type for the given predictions and true labels.

        Parameters
        ----------
        yhat : np.array
            The predicted labels or scores from the model.
        ytrue : np.array
            The true labels of the data.

        Returns
        -------
        float
            The computed accuracy value.

        Notes
        -----
        For binary and multilabel classification, labels or scores above the threshold are considered
        as class 1, and those below or equal to the threshold are considered as class 0.
        """
        self._check_shape(yhat, ytrue)
        yhat_labels = (yhat > self.threshold).astype(int)
        true_positives = np.sum((yhat_labels == 1) & (ytrue == 1))
        false_positives = np.sum((yhat_labels == 1) & (ytrue == 0))
        true_negatives = np.sum((yhat_labels == 0) & (ytrue == 0))
        false_negatives = np.sum((yhat_labels == 0) & (ytrue == 1))
        
        return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) != 0 else 0