from skynet_ml.metrics.base import BaseMetric
import numpy as np



class Accuracy(BaseMetric):
    """
    Accuracy metric for classification tasks.
    """    

    
    def __init__(self, threshold: float = 0.5, task_type: str = "binary") -> None:
        """
        Initialize the metric.

        Args:
            threshold (float, optional): Threshold used to converto probabilty into 1 or 0 . Defaults to 0.5.
            task_type (str, optional): Type of task, either binary, multilabel or multiclass. Defaults to "binary".
        """        
        
        self.threshold = threshold
        self.task_type = task_type
        self.name = f"accuracy_{str(threshold)}_{str(task_type)}"
        
        
        
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the accuracy of the predictions.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Raises:
            ValueError: If the task type is not one of binary, multilabel or multiclass.

        Returns:
            float: The accuracy of the predictions.
        """        
        
        self.check_shape(y_true, y_pred)
        
        if self.task_type == "binary":
            y_pred_labels = (y_pred > self.threshold).astype(int)
            return np.sum(y_true == y_pred_labels) / len(y_true)
            
            
        elif self.task_type == "multilabel":
            y_pred_labels = (y_pred > self.threshold).astype(int)
            return np.mean(np.sum(y_true == y_pred_labels, axis=0) / len(y_true))
            
            
        elif self.task_type == "multiclass": 
            y_pred_labels = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
            correct_predictions = np.all(y_true == y_pred_labels, axis=1)  
            return np.mean(correct_predictions) 
        
        
        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
