from skynet_ml.metrics.base import BaseMetric
import numpy as np



class Precision(BaseMetric):


    
    def __init__(self, threshold: float = 0.5, task_type: str = "binar") -> None:
        """
        Initialize the metric.

        Args:
            threshold (float, optional): Threshold used to converto probabilty into 1 or 0 . Defaults to 0.5.
            task_type (str, optional): Type of task, either binary, multilabel or multiclass. Defaults to "binary".
        """ 
        self.threshold = threshold
        self.task_type = task_type
        self.name = f"precision_{str(threshold)}_{str(task_type)}"
        
        
        
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the precision of the predictions.
        """

        self.check_shape(y_true, y_pred)
        
        if self.task_type == "binary":
            
            y_pred_labels = (y_pred > self.threshold).astype(int)
            TP = np.sum((y_true == 1) & (y_pred_labels == 1))
            FP = np.sum((y_true == 0) & (y_pred_labels == 1))
            
            return TP / (TP + FP) if (TP + FP) > 0 else 0
    
    
        elif self.task_type == "multilabel":
            
            y_pred_labels = (y_pred > self.threshold).astype(int)
            TP = np.mean(np.sum((y_true == 1) & (y_pred_labels == 1), axis=0))
            FP = np.mean(np.sum((y_true == 0) & (y_pred_labels == 1), axis=0))
            
            return TP / (TP + FP) if (TP + FP) > 0 else 0
        

        elif self.task_type == "multiclass":
            
            y_pred_labels = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
            precisions = []
            
            for class_label in range(y_pred_labels.shape[1]):  # iterate over each class
                TP = np.sum((y_true[:, class_label] == 1) & (y_pred_labels[:, class_label] == 1))  # true positive for current class
                FP = np.sum((y_true[:, class_label] == 0) & (y_pred_labels[:, class_label] == 1))  # false positive for current class
                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # calculate precision for current class, handle division by zero
                precisions.append(precision)
                
            return np.round(np.mean(precisions), decimals=4)  # return average precision across all classes
            

        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
