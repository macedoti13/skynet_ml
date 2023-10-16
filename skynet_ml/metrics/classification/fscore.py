from skynet_ml.metrics.classification.precision import Precision
from skynet_ml.metrics.classification.recall import Recall
from skynet_ml.metrics.base import BaseMetric
import numpy as np



class FScore(BaseMetric):
    """
    F-score metric for classification tasks.
    """    
   
    def __init__(self, threshold: float = 0.5, task_type: str = "binary", f: int = 1) -> None:
        """
        Initialize the metric.
        """
        
        self.precision = Precision(threshold=threshold, task_type=task_type)
        self.recall = Recall(threshold=threshold, task_type=task_type)
        self.task_type = task_type
        self.f = f
        self.name = f"fscore_{str(threshold)}_{str(task_type)}_{str(f)}"
        
        
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the F-score of the predictions.
        """        
        
        precision = self.precision.compute(y_true=y_true, y_pred=y_pred)
        recall = self.recall.compute(y_true=y_true, y_pred=y_pred)
        denominator = ((self.f**2 * precision) + recall)
        
        return np.round((1 + self.f**2) * ((precision * recall)) / denominator, decimals=4) if denominator != 0 else 0
    