from skynet_ml.metrics.metric import Metric
import numpy as np


class Recall(Metric):
    """
    Recall metric class for evaluating classification models.

    The Recall class computes the recall or sensitivity score for binary, 
    multiclass, and multilabel classification tasks. Recall is a measure 
    of a classifier's completeness. The recall is the ratio of the number of 
    true positives divided by the number of true positives plus the number 
    of false negatives.

    For `binary` classification:
    - Predictions are thresholded at the specified value.
    - Recall is computed as TP / (TP + FN).

    For `multiclass` classification:
    - The class with the highest predicted probability is considered as the 
      predicted class for each sample.
    - Micro-average recall is computed.

    For `multilabel` classification:
    - Recall is computed for each label and then averaged across all labels.

    Parameters
    ----------
    threshold : float, optional, default=0.5
        Decision threshold for classification predictions, used for binary 
        and multilabel classification tasks.

    task_type : str, default='binary'
        Specifies the type of classification task ('binary', 'multiclass', 
        'multilabel').

    Attributes
    ----------
    name : str
        Name of the metric, 'recall'.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute and return the recall of classification based on the true 
        and predicted labels.

    Example
    -------
    >>> from skynet_ml.metrics import Recall
    >>> y_true = np.array([1, 0, 1, 0, 1])
    >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
    >>> recall = Recall(threshold=0.5, task_type='binary')
    >>> recall_score = recall.compute(y_true, y_pred)
    >>> print(recall_score)
    1.0

    Notes
    -----
    - Ensure `y_true` and `y_hat` are of the same shape.
    - For multiclass and multilabel tasks, `y_hat` should contain raw predicted 
      probabilities or scores for each class.
    """
    
    
    def __init__(self, threshold: float = 0.5, task_type: str = "binary") -> None:
        """
        Initialization method for the Recall class, setting the threshold,
        task type, and name of the metric.

        Parameters
        ----------
        threshold : float
            The decision threshold for turning predicted probabilities into
            binary class predictions.
        task_type : str
            The classification task type, either 'binary', 'multiclass', 
            or 'multilabel'.

        Sets
        ----
        name : str
            Name of the metric, initialized as 'recall'.
        """
        
        self.threshold = threshold
        self.task_type = task_type
        self.name = "recall"
        
        
    def get_config(self) -> dict:
        """
        Return the configuration of the metric.

        Returns
        -------
        dict
            Configuration of the metric.
        """
        
        return {"threshold": self.threshold, "task_type": self.task_type}
        
        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the recall for the given true and predicted labels.

        The method computes recall for the provided true and predicted labels
        based on the task type specified during the object's initialization.
        Ensure y_true and y_hat have the same shape before calling this method.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels.
        y_hat : np.array
            Estimated target labels as returned by a classifier.

        Returns
        -------
        float
            Recall of the provided predictions.

        Raises
        ------
        ValueError
            If an unsupported task_type is provided.

        Example
        -------
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
        >>> recall_obj = Recall(threshold=0.5, task_type='binary')
        >>> recall_score = recall_obj.compute(y_true, y_pred)
        >>> print(recall_score)
        1.0
        """
        
        # Check the shape of the inputs
        self.check_shape(y_true, y_hat)
        
        # Calculate recall for binary tasks
        if self.task_type == "binary":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            TP = np.sum((y_true == 1) & (y_hat_labels == 1))
            FN = np.sum((y_true == 1) & (y_hat_labels == 0))
            return TP / (TP + FN)
        
        # Calculate recall for multilabel tasks
        elif self.task_type == "multilabel":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            TP = np.mean(np.sum((y_true == 1) & (y_hat_labels == 1), axis=0))
            FN = np.mean(np.sum((y_true == 1) & (y_hat_labels == 0), axis=0))
            print(TP / (TP + FN))
        
        # Calculate recall for multiclass tasks
        elif self.task_type == "multiclass":
            y_hat_labels = (y_hat == y_hat.max(axis=1)[:,None]).astype(int)
            recalls = []
            
            for class_label in range(y_hat_labels.shape[1]):  # iterate over each class
                TP = np.sum((y_true[:, class_label] == 1) & (y_hat_labels[:, class_label] == 1))  # true positive for current class
                FN = np.sum((y_true[:, class_label] == 1) & (y_hat_labels[:, class_label] == 0))  # false negative for current class
                
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # calculate recall for current class, handle division by zero
                recalls.append(recall)
                
            return np.round(np.mean(recalls), decimals=4)  # return average recall across all classes
        
        # Raise error for unknown task types
        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
