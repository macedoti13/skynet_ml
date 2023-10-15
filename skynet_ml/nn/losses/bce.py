from skynet_ml.utils.factories import ActivationsFactory
from skynet_ml.nn.losses.base import BaseLoss
import numpy as np


class BinaryCrossEntropy(BaseLoss):
    """
    Implements the Binary Cross-Entropy (BCE) loss function.
    
    BCE is used for binary classification tasks. It quantifies the difference between two probability distributions:
    the true label distribution and the predicted label distribution.
    
    Mathematically, for true labels y_true and predicted probabilities y_pred, it is defined as:

        BCE = -Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

    Notes:
        - The output values of y_pred should lie between [0, 1].
        - BCE will be high if the predicted probabilities diverge from the true labels and low if they are close.
        - It is sensitive to the confidence of the predictions, i.e., predictions that are far off from the true label 
          have a larger impact than those that are close.

    Args:
        from_logits (bool): If True, the predicted values are expected to be logits and will be passed through the sigmoid activation.
                            If False, they are expected to already be in the range [0, 1]. Default is True.
    """
    
    
    def __init__(self, from_logits: bool = True) -> None:
        """
        Initializes the loss function with its name and the `from_logits` parameter.
        """
        self.from_logits = from_logits
        self.name = f"binary_crossentropy_{str(from_logits)}"
    
        
    def compute(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Computes the Binary Cross-Entropy loss for the given true labels and predicted probabilities.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, 1).
            y_pred (np.array): Predicted probabilities from the model. Expected to have the same shape as y_true.

        Returns:
            float: The computed Binary Cross-Entropy loss.
        """
        self._check_shape(y_true, y_pred)
        
        if self.from_logits:
            sigmoid = ActivationsFactory().get_object("sigmoid")
            y_pred = sigmoid.compute(y_pred)
            
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    
    def gradient(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the gradient of the Binary Cross-Entropy loss with respect to the predicted probabilities.

        Args:
            y_true (np.array): Ground truth labels. Expected to be a 2D array with shape (batch_size, 1).
            y_pred (np.array): Predicted probabilities from the model. Expected to have the same shape as y_true.

        Returns:
            np.array: Gradient of the loss with respect to the predicted probabilities. Expected to have the same shape as y_pred.
        """
        self._check_shape(y_true, y_pred)
        
        if self.from_logits:
            sigmoid = ActivationsFactory().get_object("sigmoid")
            y_pred = sigmoid.compute(y_pred)
            
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Avoid log(0)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
