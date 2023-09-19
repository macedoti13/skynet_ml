from skynet_ml.metrics.metric import Metric
import numpy as np

class PrecisionMetric(Metric):
    """
    Precision metric for classification tasks.

    Calculates the precision for binary, multilabel, or multiclass classification tasks based on
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
        Compute the precision based on the task type for the given predictions and true labels.
    """

    def __init__(self, threshold: float = 0.5, task_type: str = "binary"):
        """
        Initialize the precision metric with the specified threshold and task type.

        Parameters
        ----------
        threshold : float, optional
            The threshold for binary and multilabel classification. Default is 0.5.

        task_type : str, optional
            The type of classification task. Default is 'binary'.
        """
        self.threshold = threshold        
        self.task_type = task_type
        self.name = "precision"
        
        if task_type not in ['binary', 'multiclass', 'multilabel']:
            raise ValueError("task_type must be one of ['binary', 'multiclass', 'multilabel']")


    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the precision for the given predictions and true labels based on the task type.

        - Binary Classification:
          For binary classification tasks, the precision is calculated based on a specified threshold.
          Predictions above the threshold are treated as positive, and those below or equal are treated as negative.

        - Multiclass Classification:
          For multiclass tasks, precision is calculated per class, and the final reported precision is 
          the average over all classes. For this scenario, `yhat` should provide class scores or probabilities 
          for each class, and `ytrue` should be in one-hot encoding format.

        - Multilabel Classification:
          For multilabel tasks, precision is calculated for each label separately. The final precision is 
          the average of the precisions for each label. In this scenario, both `yhat` and `ytrue` should be binary 
          matrices, where each column corresponds to a label, and the threshold is applied to each label separately.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities or class scores. For binary tasks, this is a 1-D array of scores or probabilities.
            For multiclass or multilabel, it's a 2-D array where each column corresponds to a class or label.

        ytrue : np.array
            True labels. For binary tasks, it's a 1-D array of true labels (0 or 1). For multiclass, it should
            be in one-hot encoding format. For multilabel, it's a binary matrix indicating the presence or absence
            of each label.

        Returns
        -------
        float
            The computed precision.

        Raises
        ------
        ValueError:
            If the shapes of `yhat` and `ytrue` are inconsistent with the specified task type.
        """
        self._check_shape(yhat, ytrue)
        if self.task_type in ['multiclass', 'multilabel'] and yhat.shape[1] == 1:
            raise ValueError("For multi-class or multi-label task type, yhat and ytrue should have more than one column.")
        
        if self.task_type == "binary":
            return self._compute_binary_precision(yhat, ytrue)
        elif self.task_type == "multiclass":
            return self._compute_multiclass_precision(yhat, ytrue)
        else:
            return self._compute_multilabel_precision(yhat, ytrue)


    def _compute_binary_precision(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute precision for binary classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities.

        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The binary precision.
        """
        yhat_labels = (yhat > self.threshold).astype(int)
        true_positives = np.sum((yhat_labels == 1) & (ytrue == 1))
        false_positives = np.sum((yhat_labels == 1) & (ytrue == 0))
        
        # Handle the case where both true positives and false positives are zero
        if true_positives == 0 and false_positives == 0:
            return 1.0

        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0


    def _compute_multiclass_precision(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute average precision for multiclass classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities.

        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The averaged multiclass precision.
        """
        n_samples, n_classes = yhat.shape
        precisions = []

        for c in range(n_classes):
            # Taking the column c for each sample
            ytrue_c = ytrue[:, c]
            yhat_c = yhat[:, c]

            # Convert probabilities to binary predictions
            yhat_binary = (yhat_c >= self.threshold).astype(int)
            
            true_positives = np.sum(yhat_binary * ytrue_c)
            predicted_positives = np.sum(yhat_binary)

            if predicted_positives == 0:
                precision_c = 1.0
            else:
                precision_c = true_positives / predicted_positives
            
            precisions.append(precision_c)

        return np.mean(precisions)


    def _compute_multilabel_precision(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute average precision for multilabel classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities.

        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The averaged multilabel precision.
        """
        n_labels = yhat.shape[1]
        precisions = []

        for l in range(n_labels):
            ytrue_l = ytrue[:, l]
            yhat_l = yhat[:, l]
            precisions.append(self._compute_binary_precision(yhat_l, ytrue_l))

        return np.mean(precisions)
