from skynet_ml.metrics.metric import Metric
import numpy as np


class Accuracy(Metric):
    """
    Accuracy metric class for evaluating classification models.

    The Accuracy class provides a way to compute the accuracy for binary, 
    multiclass, and multilabel classification tasks. It serves as a simple
    but crucial metric for evaluating the performance of classification models.
    The class requires ground truth labels and model predictions as input and 
    returns the computed accuracy as a float.

    For `binary` classification:
    - The predictions are thresholded at the specified threshold value: values 
      above or equal to the threshold are converted to 1, and values below the 
      threshold are converted to 0.
    - The accuracy is then calculated as the ratio of correctly predicted samples 
      to total samples.

    For `multiclass` classification:
    - The class with the highest predicted probability is considered as the 
      predicted class for each sample.
    - The accuracy is computed as the ratio of correctly predicted class labels 
      to total samples.

    For `multilabel` classification:
    - Each label's prediction is thresholded at the specified threshold value.
    - Micro-average accuracy is computed, considering each label's prediction
      independently, and then averaged over all labels.

    Parameters
    ----------
    threshold : float, optional, default=0.5
        The decision threshold for classification prediction. Primarily used 
        for binary and multilabel classification tasks to convert predicted 
        probabilities into class labels.
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
        Name of the metric, 'accuracy'.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float
        Compute and return the accuracy of classification based on the true and 
        predicted labels.

    Example
    -------
    >>> from skynet_ml.metrics import Accuracy
    >>> y_true = np.array([[1], [0], [1], [0], [1]])
    >>> y_pred = np.array([[0.8], [0.4], [0.9], [0.35], [0.7]])
    >>> acc = Accuracy(threshold=0.5, task_type='binary')
    >>> accuracy = acc.compute(y_true, y_pred)
    >>> print(accuracy)
    1.0

    Notes
    -----
    - Ensure `y_true` and `y_hat` are of the same shape.
    - For multiclass tasks, `y_hat` should contain raw predicted probabilities
      or scores for each class.
    - The `compute` method will raise a ValueError for unknown task types.
    """
    
    
    def __init__(self, threshold: float = 0.5, task_type: str = "binary") -> None:
        """
        Initialize Accuracy object with specified threshold and task type.

        Parameters
        ----------
        threshold : float, optional, default=0.5
            Decision threshold for classification prediction, primarily used for binary and multilabel tasks.
        task_type : str, default='binary'
            Type of classification task. Options: 'binary', 'multiclass', 'multilabel'.
        """
        
        self.threshold = threshold
        self.task_type = task_type
        self.name = "accuracy"
        
        
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
        Compute accuracy based on true and predicted labels.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a classifier.

        Returns
        -------
        float
            Accuracy of classification.
        """
        
        # Check the shape of the inputs
        self.check_shape(y_true, y_hat)
        
        # calculate binary accuracy
        if self.task_type == "binary":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            return np.sum(y_true == y_hat_labels) / len(y_true)
            
        # calculate multilabel micro average accuracy
        elif self.task_type == "multilabel":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            return np.mean(np.sum(y_true == y_hat_labels, axis=0) / len(y_true))
            
        # calculate multiclass accuracy
        elif self.task_type == "multiclass": 
            y_hat_labels = (y_hat == y_hat.max(axis=1)[:,None]).astype(int)
            correct_predictions = np.all(y_true == y_hat_labels, axis=1)  # Compare rows, not individual elements
            return np.mean(correct_predictions)  # Calculate accuracy as mean of correct predictions
        
        # raise error if task type is unknown
        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
