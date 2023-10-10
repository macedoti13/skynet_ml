from skynet_ml.nn.losses.loss import Loss
from skynet_ml.utils.factories import ActivationsFactory
import numpy as np


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy (BCE) Loss Class

    This class represents the BCE loss, used in binary classification tasks. It computes the average negative log 
    likelihood between true class labels and predicted probabilities. The `compute` method calculates the loss, while 
    the `gradient` method computes the gradient of the loss with respect to the predictions.

    When from_logits=True, it assumes that the predicted values (y_hat) are logits (i.e., the output of a linear layer),
    and hence it would apply a sigmoid function to convert them into probabilities. This means that when from_logits=True,
    the output layer of the network MUST be a linear layer. If from_logits is set to True and the network's output layer 
    is a sigmoid, the sigmoid function would be applied twice, resulting in incorrect results.

    When from_logits=False, it assumes y_hat to be probabilities and uses them directly to compute the loss. In this case,
    the output layer of the network should be a sigmoid layer to ensure that the predictions are probabilities.

    Methods
    -------
    compute(y_true: np.array, y_hat: np.array) -> float:
        Computes the binary cross entropy loss.

    gradient(y_true: np.array, y_hat: np.array) -> np.array:
        Computes the gradient of the binary cross entropy loss with respect to the predictions.
    """


    def __init__(self, from_logits: bool = True) -> None:
        """
        Parameters
        ----------
        from_logits : bool, optional
            Whether the predicted values are probabilities or logits.
            If True, it applies sigmoid on predicted values to get probabilities.
            If False, it uses predicted values as probabilities.
        """
        self.from_logits = from_logits
        self.name = "binary_crossentropy"
        
        
    def compute(self, y_true: np.array, y_hat: np.array) -> float:
        """
        Compute the binary cross entropy loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        float
            The computed binary cross entropy loss value.
        """
        self._check_shape(y_true, y_hat)
        
        if self.from_logits:
            y_hat = ActivationsFactory.get_object("sigmoid").compute(y_hat)
            
        epsilon = 1e-7
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon) # Avoid log(0)

        return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
    
    
    def gradient(self, y_true: np.array, y_hat: np.array) -> np.array:
        """
        Compute the gradient of the binary cross entropy loss.

        Parameters
        ----------
        y_true : np.array
            Ground truth (correct) target values.
        y_hat : np.array
            Estimated targets as returned by a model.
            
        Returns
        -------
        np.array
            The gradient of the binary cross entropy loss with respect to predictions.
        """
        self._check_shape(y_true, y_hat)
        
        if self.from_logits:
            y_hat = ActivationsFactory.get_object("sigmoid").compute(y_hat)
            
        epsilon = 1e-7
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return (y_hat - y_true) / (y_hat * (1 - y_hat))
        
    
    def get_config(self) -> dict:
        """
        Get the configuration of the binary cross entropy loss.

        Returns
        -------
        dict
            The configuration of the binary cross entropy loss.
        """
        return {"from_logits": self.from_logits}
