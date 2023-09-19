from skynet_ml.metrics.metric import Metric
from skynet_ml.metrics.precision import PrecisionMetric
from skynet_ml.metrics.recall import RecallMetric
import numpy as np

class FScoreMetric(Metric):
    """
    Compute the F-score for classification tasks.

    The F-score is a measure of a test's accuracy, considering both precision and recall. 
    It is the harmonic mean of precision and recall, with a higher score as a better result.

    Attributes
    ----------
    f : int
        The factor that determines the weight of precision in the combined score.
    threshold : float
        Threshold value for binary and multilabel classification. Values above this threshold
        are considered as class 1, and values below or equal are considered as class 0.
        
    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the F-score for the given predictions and true labels.
    """

    def __init__(self, threshold: float = 0.5, f: int = 1, task_type: str = "binary") -> None:
        """
        Initialize the F-score metric with the specified threshold and f factor.

        Parameters
        ----------
        threshold : float, optional
            The threshold for binary classification. Default is 0.5.

        f : int, optional
            The factor determining the weight of precision. Default is 1.
        """
        self.recall = RecallMetric(threshold, task_type)
        self.precision = PrecisionMetric(threshold, task_type)
        self.f = f
        self.name = "fscore"


    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the F-score for the given predictions and true labels based on the task type.

        The F-score, also known as the F-measure, is a harmonic mean of precision and recall. 
        It provides a single score that balances the trade-off between precision and recall.

        Specifically, the F-score is defined as:
        F = (1 + f^2) * (precision * recall) / (f^2 * precision + recall)

        where:
        - precision is the ratio of true positive predictions to the total predicted positives.
        - recall (or sensitivity) is the ratio of true positive predictions to the actual number of positives.
        - f is a factor that determines the weight of precision in the combined score. 
          When f=1, this is commonly known as the F1-score, which gives equal weight to precision and recall.

        - Binary Classification:
          The predictions are thresholded to produce binary labels. The F-score is then computed using 
          these binary labels and the true labels.

        - Multilabel Classification:
          The F-score is computed for each label and then averaged to produce an overall score. 
          Each label's predictions are thresholded to produce binary predictions for that label.

        - Multiclass Classification:
          The predicted class is the one with the highest score. The F-score is computed by treating 
          each class as the positive class and all others as the negative class, then averaging over all classes.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities or class scores. 
            For binary and multilabel tasks, this can be a 1-D or 2-D array, respectively.
            For multiclass, it's a 2-D array where each column corresponds to a class's score or probability.

        ytrue : np.array
            True labels. For binary tasks, it's a 1-D array of true labels (0 or 1). 
            For multilabel tasks, it's a 2-D binary array indicating the presence or absence of each label.
            For multiclass, it's a 2-D one-hot encoded array where each row indicates the true class.

        Returns
        -------
        float
            The computed F-score, ranging between 0 (worst) and 1 (best).

        Raises
        ------
        ValueError:
            If the shapes of `yhat` and `ytrue` are inconsistent with the expected input shapes.
        """
        precision = self.precision.compute(yhat, ytrue)
        recall = self.recall.compute(yhat, ytrue)
        
        # Check for zero denominator
        if (self.f**2 * precision + recall) == 0:
            return 0.0

        return (1 + self.f**2) * ((precision * recall) / ((self.f**2 * precision) + recall))
