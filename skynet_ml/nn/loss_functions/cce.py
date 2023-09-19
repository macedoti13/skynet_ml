from skynet_ml.nn.loss_functions.loss import Loss
import numpy as np

class CrossEntropyLoss(Loss):
    """
    Loss function that calculates the categorical cross-entropy between predicted probabilities and one-hot encoded true labels.

    The categorical cross-entropy measures the performance of a multi-class classification model. The value is 
    0 if the predicted probabilities match perfectly with the true labels, and increases as the predictions 
    deviate from the true labels.

    Attributes
    ----------
    with_softmax : bool
        Flag to indicate if the softmax activation was applied to the logits. If True, the gradient is simply 
        the difference between predicted probabilities and true labels.

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the categorical cross-entropy value for given predictions and true labels.

    gradient(yhat: np.array, ytrue: np.array) -> np.array:
        Compute the gradient of the categorical cross-entropy with respect to the predictions for given 
        predictions and true labels.
    """
    
    def __init__(self, with_softmax: bool = True) -> None:
        """
        Initializes a new instance of the CrossEntropyLoss class.

        Parameters
        ----------
        with_softmax : bool, optional
            Flag to indicate if softmax was applied to logits, by default True.

        """
        self.with_softmax = with_softmax
        
        
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the categorical cross-entropy value for given predictions and true labels.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities for each class.
        
        ytrue : np.array
            One-hot encoded true labels.

        Returns
        -------
        float
            The computed categorical cross-entropy value.

        """
        self._check_shape(yhat, ytrue)
        
        epsilon = 1e-15
        yhat = np.clip(yhat, epsilon, 1 - epsilon) # Avoid log(0)
        
        return -np.mean(np.sum(ytrue * np.log(yhat), axis=1, keepdims=True))
    
    
    def gradient(self, yhat: np.array, ytrue: np.array) -> np.array:
        """
        Compute the gradient of the categorical cross-entropy with respect to the predictions.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities for each class.
        
        ytrue : np.array
            One-hot encoded true labels.

        Returns
        -------
        np.array
            The gradient of the categorical cross-entropy with respect to the predictions.

        """
        self._check_shape(yhat, ytrue)
        
        epsilon = 1e-15
        yhat = np.clip(yhat, epsilon, 1 - epsilon) # Avoid division by zero
        
        if self.with_softmax:
            return yhat - ytrue
        
        return - (ytrue / yhat)
