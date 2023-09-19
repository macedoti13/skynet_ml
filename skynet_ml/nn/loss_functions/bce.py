from skynet_ml.nn.loss_functions.loss import Loss
import numpy as np

class BinaryCrossEntropyLoss(Loss):
    """
    Loss function that calculates the binary cross-entropy (BCE) between predicted and true binary labels.

    The BCE measures the performance of a binary classification model. The value is 
    0 if the predicted and true values match perfectly, and increases as the predictions 
    deviate from the true values.

    Attributes
    ----------
    None

    Methods
    -------
    compute(yhat: np.array, ytrue: np.array) -> float:
        Compute the BCE value for given predictions and true labels.

    gradient(yhat: np.array, ytrue: np.array) -> np.array:
        Compute the gradient of the BCE with respect to the predictions for given 
        predictions and true labels.
    """
    
    def compute(self, yhat: np.array, ytrue: np.array) -> float:
        """
        Compute the BCE value for given predictions and true labels.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities (between 0 and 1).
        
        ytrue : np.array
            True binary labels (0 or 1).

        Returns
        -------
        float
            The computed BCE value.

        """
        self._check_shape(yhat, ytrue)
        
        epsilon = 1e-15 
        yhat = np.clip(yhat, epsilon, 1 - epsilon) # Avoid log(0)
        
        return -np.mean(ytrue * np.log(yhat) + (1 - ytrue) * np.log(1 - yhat))
    
    
    def gradient(self, yhat: np.array, ytrue: np.array) -> np.array:
        """
        Compute the gradient of the BCE with respect to the predictions.

        Parameters
        ----------
        yhat : np.array
            Predicted probabilities (between 0 and 1).
        
        ytrue : np.array
            True binary labels (0 or 1).

        Returns
        -------
        np.array
            The gradient of the BCE with respect to the predictions.

        """
        self._check_shape(yhat, ytrue)
        
        epsilon = 1e-15
        yhat = np.clip(yhat, epsilon, 1 - epsilon) # Avoid division by zero
        
        return (yhat - ytrue) / (yhat * (1 - yhat))
