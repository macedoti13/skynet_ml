from skynet_ml.metrics.metric import Metric
import numpy as np

class RecallMetric(Metric):
    """
    Compute the recall for classification tasks.

    Given predictions and true labels, this metric calculates the recall for binary,
    multiclass, or multilabel classification tasks based on the specified task type.

    Attributes
    ----------
    threshold : float
        Threshold value for binary classification. Values above this threshold
        are considered positive (class 1), otherwise negative (class 0).

    task_type : str
        The type of classification task. Must be one of 'binary', 'multiclass', or 'multilabel'.
    """

    def __init__(self, threshold: float = 0.5, task_type: str = "binary"):
        """
        Initialize the recall metric with the specified threshold and task type.

        Parameters
        ----------
        threshold : float, optional
            The threshold for binary classification. Default is 0.5.

        task_type : str, optional
            The type of classification task. Default is 'binary'.
        """
        self.threshold = threshold        
        self.task_type = task_type
        self.name = "recall"
        
        if task_type not in ['binary', 'multiclass', 'multilabel']:
            raise ValueError("task_type must be one of ['binary', 'multiclass', 'multilabel']")


    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the recall for the given predictions and true labels based on the task type.

        - Binary Classification:
          For binary classification tasks, the recall is calculated based on a specified threshold.
          Predictions above the threshold are treated as positive, otherwise as negative.

        - Multiclass Classification:
          For multiclass tasks, recall is calculated per class and the final reported recall is 
          the macro-average over all classes. For this scenario, `yhat` should provide class scores
          or probabilities for each class and `ytrue` should be in one-hot encoding format.

        - Multilabel Classification:
          For multilabel tasks, recall is calculated for each label separately. The final recall is 
          the macro-average of the recalls for each label. In this scenario, both `yhat` and `ytrue`
          should be binary matrices, where each column corresponds to a label.

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
            The computed recall.

        Raises
        ------
        ValueError:
            If the shapes of `yhat` and `ytrue` are inconsistent with the specified task type.
        """
        self._check_shape(yhat, ytrue)
        if self.task_type in ['multiclass', 'multilabel'] and yhat.shape[1] == 1:
            raise ValueError("For multi-class or multi-label task type, yhat and ytrue should have more than one column.")
        
        if self.task_type == "binary":
            return self._compute_binary_recall(yhat, ytrue)
        elif self.task_type == "multiclass":
            return self._compute_multiclass_recall(yhat, ytrue)
        else:
            return self._compute_multilabel_recall(yhat, ytrue)
        
        
    def _compute_binary_recall(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute recall for binary classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities for the positive class.

        ytrue : np.array
            True labels.

        Returns
        -------
        float
            The computed binary recall.
        """
        yhat_labels = (yhat > self.threshold).astype(int)
        true_positives = np.sum((yhat_labels == 1) & (ytrue == 1))
        false_negatives = np.sum((yhat_labels == 0) & (ytrue == 1))
        
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    
    
    def _compute_multiclass_recall(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute macro-average recall for multiclass classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities for each class.

        ytrue : np.array
            True labels in one-hot encoding.

        Returns
        -------
        float
            The computed multiclass recall.
        """
        n_classes = yhat.shape[1]
        recalls = []
        
        for c in range(n_classes):
            ytrue_c = ytrue[:, c]
            yhat_c = (np.argmax(yhat, axis=1) == c).astype(int)
            recalls.append(self._compute_binary_recall(yhat_c, ytrue_c))

        return np.mean(recalls)
    
    
    def _compute_multilabel_recall(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute macro-average recall for multilabel classification.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities or scores for each label.

        ytrue : np.array
            True binary labels for each label.

        Returns
        -------
        float
            The computed multilabel recall.
        """
        n_labels = yhat.shape[1]
        recalls = []
        
        for l in range(n_labels):
            ytrue_l = ytrue[:, l]
            yhat_l = yhat[:, l]
            recalls.append(self._compute_binary_recall(yhat_l, ytrue_l))

        return np.mean(recalls)
