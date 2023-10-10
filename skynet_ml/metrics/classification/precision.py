from skynet_ml.metrics.metric import Metric
import numpy as np


class Precision(Metric):
    """
    Precision metric class for evaluating classification models.

    The Precision class computes the precision score for binary, multiclass, 
    and multilabel classification tasks. Precision is a measure of a classifier's 
    exactness or quality, calculated as the ratio of true positives to the sum 
    of true and false positives. This metric is crucial when the costs of false 
    positives are high.

    For `binary` classification:
    - Predictions are thresholded at the specified threshold value.
    - Precision is computed as TP / (TP + FP), where TP is the number of true 
      positives and FP is the number of false positives.

    For `multiclass` classification:
    - The class with the highest predicted probability is considered as the 
      predicted class for each sample.
    - Micro-average precision is computed.

    For `multilabel` classification:
    - Precision is computed for each label, and the average precision score is 
      returned.

    Parameters
    ----------
    threshold : float, optional, default=0.5
        Decision threshold for classification prediction. Used for binary and 
        multilabel classification tasks to convert predicted probabilities into 
        class labels.
        For binary and multilabel tasks, values >= threshold are assigned 1, 
        and values < threshold are assigned 0.

    task_type : str, default='binary'
        Specifies the type of classification task.
        Options:
        - 'binary': for binary classification tasks.
        - 'multiclass': for multiclass classification tasks.
        - 'multilabel': for multilabel classification tasks.

    Attributes
    ----------
    name : str
        Name of the metric, 'precision'.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute and return the precision of classification based on the true and 
        predicted labels.

    Example
    -------
    >>> from skynet_ml.metrics import Precision
    >>> y_true = np.array([1, 0, 1, 0, 1])
    >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
    >>> precision = Precision(threshold=0.5, task_type='binary')
    >>> precision_score = precision.compute(y_true, y_pred)
    >>> print(precision_score)
    1.0

    Notes
    -----
    - Ensure `y_true` and `y_hat` are of the same shape.
    - For multiclass and multilabel tasks, `y_hat` should contain raw predicted 
      probabilities or scores for each class.
    - The `compute` method will raise a ValueError for unknown task types.
    - Precision is sensitive to class imbalance and might not reflect the model's 
      performance accurately if the negative class significantly outnumbers the 
      positive class.
    """
    
    
    def __init__(self, threshold: float = 0.5, task_type: str = "binar") -> None:
        """
        Initialize a Precision object with specified threshold and task type.

        Parameters
        ----------
        threshold : float, optional, default=0.5
            The decision threshold for classification prediction. Used primarily 
            for binary and multilabel classification tasks to convert predicted 
            probabilities into class labels.
        task_type : str, default='binary'
            Specifies the type of classification task. Options include 'binary', 
            'multiclass', and 'multilabel'.

        Attributes
        ----------
        name : str
            Name of the metric initialized as 'precision'.
        """
        
        self.threshold = threshold
        self.task_type = task_type
        self.name = "precision"
        
        
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
        Compute precision for the given true and predicted labels.

        This method computes the precision score for the provided true and 
        predicted labels based on the task type specified during the object's 
        initialization.

        For binary tasks, precision is computed as TP / (TP + FP).
        For multilabel tasks, precision is computed for each label individually, 
        then averaged.
        For multiclass tasks, micro-averaged precision is computed.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels.
        y_hat : np.array
            Estimated target labels as returned by a classifier.

        Returns
        -------
        float
            Precision of the provided predictions.

        Raises
        ------
        ValueError
            If an unsupported task_type is provided.

        Example
        -------
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
        >>> precision_obj = Precision(threshold=0.5, task_type='binary')
        >>> precision_score = precision_obj.compute(y_true, y_pred)
        >>> print(precision_score)
        1.0

        Notes
        -----
        - Ensure `y_true` and `y_hat` have the same shape.
        - For multiclass and multilabel tasks, `y_hat` should contain raw 
          predicted probabilities or decision function scores for each class.
        """
        
        # Check the shape of the inputs
        self.check_shape(y_true, y_hat)
        
        # Calculate precision for binary tasks
        if self.task_type == "binary":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            TP = np.sum((y_true == 1) & (y_hat_labels == 1))
            FP = np.sum((y_true == 0) & (y_hat_labels == 1))
            return TP / (TP + FP)
        
        # Calculate precision for multilabel tasks
        elif self.task_type == "multilabel":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            TP = np.mean(np.sum((y_true == 1) & (y_hat_labels == 1), axis=0))
            FP = np.mean(np.sum((y_true == 0) & (y_hat_labels == 1), axis=0))
            return TP / (TP + FP)
        
        # Calculate precision for multiclass tasks
        elif self.task_type == "multiclass":
            y_hat_labels = (y_hat == y_hat.max(axis=1)[:,None]).astype(int)
            precisions = []
            
            for class_label in range(y_hat_labels.shape[1]):  # iterate over each class
                TP = np.sum((y_true[:, class_label] == 1) & (y_hat_labels[:, class_label] == 1))  # true positive for current class
                FP = np.sum((y_true[:, class_label] == 0) & (y_hat_labels[:, class_label] == 1))  # false positive for current class
                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # calculate precision for current class, handle division by zero
                precisions.append(precision)
                
            return np.round(np.mean(precisions), decimals=4)  # return average precision across all classes
            
        # Raise error for unknown task types
        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
