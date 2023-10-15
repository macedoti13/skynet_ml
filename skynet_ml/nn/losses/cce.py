from skynet_ml.utils.factories import ActivationsFactory
from skynet_ml.nn.losses.base import BaseLoss
import numpy as np


class CategoricalCrossEntropy(BaseLoss):
    """
    Implements the Categorical Cross-Entropy (CCE) loss function.
    
    CCE is used for multi-class classification tasks. It quantifies the difference between two probability distributions:
    the true label distribution and the predicted label distribution.
    
    Mathematically, for true labels y_true and predicted probabilities y_pred, it is defined as:

        CCE = -Σ [y_true * log(y_pred)]

    Notes:
        - The output values of y_pred for each sample should sum up to 1.
        - CCE will be high if the predicted probabilities diverge from the true labels and low if they are close.
        - Like Binary Cross-Entropy, it is sensitive to the confidence of the predictions.

    Args:
        from_logits (bool): If True, the predicted values are expected to be logits and will be passed through the softmax activation.
                            If False, they are expected to sum up to 1 for each sample. Default is True.
    """
    
    
    def __init__(self, from_logits: bool = True) -> None:
        """
        Initializes the loss function with its name and the `from_logits` parameter.
        """
        self.from_logits = from_logits
        self.name = f"categorical_crossentropy_{str(from_logits)}"
        
        
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the Categorical Cross-Entropy loss for the given true labels and predicted probabilities.

        Args:
            y_true (np.array): Ground truth labels in one-hot encoded format. 
                               Expected to be a 2D array with shape (batch_size, num_classes).
            y_pred (np.array): Predicted probabilities from the model. Expected to have the same shape as y_true.

        Returns:
            float: The computed Categorical Cross-Entropy loss.
        """
        self._check_shape(y_true, y_pred)
        
        if self.from_logits:
            softmax = ActivationsFactory().get_object("softmax")
            y_pred = softmax.compute(y_pred)
            
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1, keepdims=True))
    
    
    def gradient(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the gradient of the Categorical Cross-Entropy loss with respect to the predicted probabilities.

        Args:
            y_true (np.array): Ground truth labels in one-hot encoded format. 
                               Expected to be a 2D array with shape (batch_size, num_classes).
            y_pred (np.array): Predicted probabilities from the model. Expected to have the same shape as y_true.

        Returns:
            np.array: Gradient of the loss with respect to the predicted probabilities. Expected to have the same shape as y_pred.
        """
        self._check_shape(y_true, y_pred)
    
        if self.from_logits:
            softmax = ActivationsFactory().get_object("softmax")
            y_pred = softmax.compute(y_pred)
        
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return y_pred - y_true
