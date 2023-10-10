from skynet_ml.metrics.metric import Metric
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ConfusionMatrix(Metric):
    """
    ConfusionMatrix metric class for evaluating classification tasks.

    The ConfusionMatrix class calculates the confusion matrix for binary, 
    multiclass, and multilabel classification tasks. It describes the 
    performance of a classification model by summarizing the frequencies 
    of correct and incorrect predictions.

    Parameters
    ----------
    threshold : float, optional, default=0.5
        Decision threshold used for binary and multilabel tasks. Predictions 
        with probability greater than or equal to threshold are treated as 
        positive class.
    
    task_type : str, default='binary'
        Classification task type. Should be one of the following: 'binary', 
        'multiclass', or 'multilabel'.

    Attributes
    ----------
    threshold : float
        Decision threshold.
    task_type : str
        Classification task type.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> np.array
        Compute the confusion matrix based on true and predicted labels.
    binary_confusion_matrix(y_true: np.array, y_hat: np.array) -> np.array
        Compute the binary classification confusion matrix.
    multilabel_confusion_matrix(y_true: np.array, y_hat: np.array) -> np.array
        Compute the multilabel classification confusion matrix.
    multiclass_confusion_matrix(y_true: np.array, y_hat: np.array) -> np.array
        Compute the multiclass classification confusion matrix.

    Example
    -------
    >>> from skynet_ml.metrics import ConfusionMatrix
    >>> y_true = np.array([1, 0, 1, 0, 1])
    >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
    >>> cm = ConfusionMatrix(threshold=0.5, task_type='binary')
    >>> matrix = cm.compute(y_true, y_pred)
    >>> print(matrix)
    array([[3, 0], [0, 2]]
    """
    
    
    def __init__(self, threshold: float = 0.5, task_type: str = "binary") -> None:
        """
        Initialize the ConfusionMatrix object with specified threshold and task type.
        """
        
        self.threshold = threshold
        self.task_type = task_type
        
        
    def get_config(self) -> dict:
        """
        Return the configuration of the metric.

        Returns
        -------
        dict
            Configuration of the metric.
        """
        
        return {"threshold": self.threshold, "task_type": self.task_type}
        
    
    def compute(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the confusion matrix for the provided ground truth and predicted labels.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels.
        y_hat : np.array
            Estimated target labels as returned by a classifier.

        Returns
        -------
        np.array
            The computed confusion matrix.

        Example
        -------
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0.8, 0.4, 0.9, 0.35, 0.7])
        >>> cm = ConfusionMatrix(threshold=0.5, task_type='binary')
        >>> matrix = cm.compute(y_true, y_pred)
        >>> print(matrix)
        array([[3, 0], [0, 2]]
        """
        
        # Check the shape of the inputs
        self.check_shape(y_true, y_hat)
        
        # calculate binary confusion matrix
        if self.task_type == "binary":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            return self.binary_confusion_matrix(y_true, y_hat_labels)
        
        # calculate multilabel confusion matrix
        elif self.task_type == "multilabel":
            y_hat_labels = (y_hat > self.threshold).astype(int)
            return self.multilabel_confusion_matrix(y_true, y_hat_labels)
        
        # calculate multiclass confusion matrix
        elif self.task_type == "multiclass":
            y_hat_labels = np.where(np.greater_equal(y_hat,  np.max(y_hat, axis=1)[:, None]), 1, 0)
            return self.multiclass_confusion_matrix(y_true, y_hat_labels)
        
        # raise error if task type is unknown
        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
        
        
    def binary_confusion_matrix(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the binary classification confusion matrix.
        
        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels for binary classification.
        y_hat : np.array
            Estimated target labels as returned by a classifier for binary classification.

        Returns
        -------
        np.array
            The computed binary classification confusion matrix.
            The matrix is of the form:
            [[TP, FP]
             [FN, TN]]

        Example
        -------
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> matrix = binary_confusion_matrix(y_true, y_pred)
        >>> print(matrix)
        array([[3, 0], [1, 2]]
        """
        
        TP = np.sum((y_hat == 1) & (y_true == 1))
        TN = np.sum((y_hat == 0) & (y_true == 0))
        FP = np.sum((y_hat == 1) & (y_true == 0))
        FN = np.sum((y_hat == 0) & (y_true == 1))

        return np.array([[TP, FP], [FN, TN]])
    
    
    def multilabel_confusion_matrix(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the multilabel classification confusion matrix.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels for multilabel classification.
        y_hat : np.array
            Estimated target labels as returned by a classifier for multilabel classification.

        Returns
        -------
        np.array
            Stack of computed confusion matrices for each label in multilabel classification.

        Example
        -------
        ... (Include a relevant example here) ...
        """
        
        TP = np.sum((y_hat == 1) & (y_true == 1), axis=0)
        TN = np.sum((y_hat == 0) & (y_true == 0), axis=0)
        FP = np.sum((y_hat == 1) & (y_true == 0), axis=0)
        FN = np.sum((y_hat == 0) & (y_true == 1), axis=0)
        
        array_list = []
        for i in range(len(TP)):
            array_list.append(np.array([[TP[i], FP[i]], [FN[i], TN[i]]]))

        return np.stack(array_list)
    
    
    def multiclass_confusion_matrix(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the multiclass classification confusion matrix.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target labels for multiclass classification.
        y_hat : np.array
            Estimated target labels as returned by a classifier for multiclass classification.

        Returns
        -------
        np.array
            The computed multiclass classification confusion matrix.

        Example
        -------
        ... (Include a relevant example here) ...
        """
        
        n_classes = y_true.shape[1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i in range(n_classes):
            for j in range(n_classes):
                conf_matrix[i, j] = np.sum(y_true[:, i] * y_hat[:, j])

        return conf_matrix


    def plot(self, cm: np.array, save_in: str = None) -> None:
        """
        Plot confusion matrix or matrices.

        If the task type is 'multilabel', iterates over each confusion matrix
        in the provided array and plots it. Otherwise, plots the single
        provided confusion matrix. The plot is saved to a file if a path is
        provided, otherwise it is shown using plt.show().

        Parameters
        ----------
        cm : np.array
            Confusion matrix or matrices to be plotted. If the task type is 
            'multilabel', this should be an array of confusion matrices.
        save_in : str, optional
            Path to save the plotted image. If None, the plot will be shown
            using plt.show() (default is None).

        Returns
        -------
        None
        """
        
        if self.task_type == "multilabel":
            for i in cm:
                self.plot_confusion_matrix(i, save_in)
        else:
            self.plot_confusion_matrix(cm, save_in)
            
            
    def plot_confusion_matrix(self, cm: np.array, save_in: str = None):
        """
        Plot a single confusion matrix.

        Sets up the plot, draws a heatmap of the confusion matrix, annotates
        it, sets labels and title, and either saves or shows the plot.

        Parameters
        ----------
        cm : np.array
            Confusion matrix to be plotted.
        save_in : str, optional
            Path to save the plotted image. If None, the plot will be shown
            using plt.show() (default is None).

        Returns
        -------
        None
        """
        # Set up the matplotlib figure
        plt.figure(figsize=(8, 6))

        # Set custom color map
        cmap = sns.diverging_palette(220, 150, as_cmap=True, center="dark")

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .75}, annot_kws={"size": 16})

        # Set labels
        plt.title('Confusion Matrix', fontsize=20)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)

        # Save plot if save_in is provided
        if save_in:
            plt.savefig(save_in)
        else:
            # Show plot if no save path is provided
            plt.show()
        