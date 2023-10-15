from skynet_ml.metrics.base import BaseMetric
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



class ConfusionMatrix(BaseMetric):
    """
    Confusion matrix for classification tasks.
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
        self.name = f"confusion_matrix_{str(threshold)}_{str(task_type)}"
        
        
    
    def compute(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the confusion matrix of the predictions.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Raises:
            ValueError: If the task type is not one of binary, multilabel or multiclass.

        Returns:
            np.array: The confusion matrix of the predictions.
        """        
        
        self.check_shape(y_true, y_pred)
        

        if self.task_type == "binary":
            y_pred_labels = (y_pred > self.threshold).astype(int)
            return self.binary_confusion_matrix(y_true, y_pred_labels)
        

        elif self.task_type == "multilabel":
            y_pred_labels = (y_pred > self.threshold).astype(int)
            return self.multilabel_confusion_matrix(y_true, y_pred_labels)
        

        elif self.task_type == "multiclass":
            y_pred_labels = np.where(np.greater_equal(y_pred,  np.max(y_pred, axis=1)[:, None]), 1, 0)
            return self.multiclass_confusion_matrix(y_true, y_pred_labels)
        

        else:
            raise ValueError(f"Unknown task type {self.task_type}.")
        
        
        
    def binary_confusion_matrix(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the binary confusion matrix of the predictions.
        """

        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        return np.array([[TP, FP], [FN, TN]])
    
    
    
    def multilabel_confusion_matrix(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the multilabel confusion matrix of the predictions.
        """

        TP = np.sum((y_pred == 1) & (y_true == 1), axis=0)
        TN = np.sum((y_pred == 0) & (y_true == 0), axis=0)
        FP = np.sum((y_pred == 1) & (y_true == 0), axis=0)
        FN = np.sum((y_pred == 0) & (y_true == 1), axis=0)
        
        array_list = []
        for i in range(len(TP)):
            array_list.append(np.array([[TP[i], FP[i]], [FN[i], TN[i]]]))

        return np.stack(array_list)
    
    
    
    def multiclass_confusion_matrix(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the multiclass confusion matrix of the predictions.
        """

        n_classes = y_true.shape[1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i in range(n_classes):
            for j in range(n_classes):
                conf_matrix[i, j] = np.sum(y_true[:, i] * y_pred[:, j])

        return conf_matrix



    def plot(self, cm: np.array, save_in: str = None) -> None:
        """
        Plots the confusion matrix.

        Args:
            cm (np.array): Confusion matrix.
            save_in (str, optional): Path where plot will be saved. Defaults to None.
        """        
        
        if self.task_type == "multilabel":
            for i in cm:
                self.plot_confusion_matrix(i, save_in)
        else:
            self.plot_confusion_matrix(cm, save_in)
            
            
            
    def plot_confusion_matrix(self, cm: np.array, save_in: str = None):
        """
        Plots the confusion matrix with seaborn.
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
        if save_in is not None:
            plt.savefig(save_in)
        else:
            # Show plot if no save path is provided
            plt.show()
