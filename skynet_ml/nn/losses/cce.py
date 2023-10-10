from skynet_ml.nn.losses.loss import Loss
from skynet_ml.utils.factories import ActivationsFactory
import numpy as np


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy (CCE) Loss Class
    
    This class is used for multi-class classification problems. It computes the average negative log likelihood
    between the true class labels (in one-hot encoded format) and the predicted probabilities. The `compute` method
    calculates the loss, while the `gradient` method computes the gradient of the loss with respect to the predictions.

    When `from_logits=True`, it assumes that the predicted values (y_hat) are logits (i.e., the output of a linear layer),
    and applies a softmax function to convert them into probabilities. Therefore, when `from_logits=True`, the network's
    output layer MUST be a linear layer, since the softmax function is applied inside the loss function. If `from_logits`
    is set to True and the network's output layer is softmax, the softmax function would be applied twice, leading to 
    incorrect results.

    When `from_logits=False`, it assumes `y_hat` are probabilities, thus, it expects the network's output layer to be 
    softmax to ensure that the predictions are probabilities. Note that the gradient simplification `y_hat - y_true` 
    is valid only when using softmax as the activation function in the output layer due to the derivative properties of 
    softmax combined with CCE.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float:
        Computes the categorical cross entropy loss.

    gradient(y_true: np.array, y_hat: np.array) -> np.array:
        Computes the gradient of the categorical cross entropy loss with respect to the predictions.
    """
     
     
    def __init__(self, from_logits: bool = True) -> None:
        """
        Initialize the Categorical Cross Entropy Loss object.
        
        Parameters
        ----------
        from_logits : bool, optional
            Flag to denote whether the network's output is logits or probabilities. 
            - True: expects logits and applies softmax function to convert logits to probabilities.
            - False: expects probabilities, assumes network's output layer is softmax. 
            Default is True.
        """
        self.from_logits = from_logits
        self.name = "categorical_crossentropy"
        
    
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the Categorical Cross Entropy loss between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array
            True class labels in one-hot encoded format.
        y_hat : np.array
            Predicted class logits or probabilities, based on the `from_logits` flag.

        Returns
        -------
        float
            The computed Categorical Cross Entropy loss.
        """
        self._check_shape(y_true, y_hat)
        
        if self.from_logits:
            softmax = ActivationsFactory().get_object("softmax")
            y_hat = softmax.compute(y_hat)
            
        epsilon = 1e-7
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        return -np.mean(np.sum(y_true * np.log(y_hat), axis=1, keepdims=True))
    
    
    def gradient(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the gradient of the Categorical Cross Entropy loss with respect to predictions.
        
        Parameters
        ----------
        y_true : np.array
            True class labels in one-hot encoded format.
        y_hat : np.array
            Predicted class logits or probabilities, based on the `from_logits` flag.
            
        Returns
        -------
        np.array
            Gradient of the Categorical Cross Entropy loss with respect to predictions.
        """
        self._check_shape(y_true, y_hat)
    
        if self.from_logits:
            softmax = ActivationsFactory().get_object("softmax")
            y_hat = softmax.compute(y_hat)
        
        epsilon = 1e-7
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return y_hat - y_true


    def get_config(self) -> dict:
        """
        Get the configuration of the categorical cross loss.

        Returns
        -------
        dict
            The configuration of the categorical cross entropy loss.
        """
        return {"from_logits": self.from_logits}
