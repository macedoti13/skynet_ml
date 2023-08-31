import numpy as np
from typing import Callable, Any

def get_loss_function(loss: str) -> Callable[..., Any]:
    """
    Returns the loss function corresponding to the provided name.
    """
    if loss == "mse":
        return mse
    elif loss == "bce":
        return binary_cross_entropy


def get_loss_derivative(loss: str) -> Callable[..., Any]:
    """
    Returns the derivative of the loss function corresponding to the provided name.
    """
    if loss == "mse":
        return mse_derivative
    elif loss == "bce":
        return binary_crossentropy_derivative
    
    
def mse(yhat: float, y: float) -> float:
    """
    Mean Squared Error Function.

    Args:
        yhat (float): Prediction.
        y (float): True label.
    """    
    return np.mean(0.5 * (yhat - y)**2)


def mse_derivative(yhat: float, y: float) -> float:
    """
    Derivative of the Mean Squared Error Function.

    Args:
        yhat (float): Prediction.
        y (float): True Label.
    """    
    return yhat - y


def binary_cross_entropy(yhat: float, y: float) -> float:
    """
    Binary Cross-Entropy Loss function. 

    Args:
        yhat (float): Prediction.
        y (float): True label.

    Returns:
        float: Loss value. Folows the formula for the BCE Loss.
    """    
    epsilon = 1e-15  # To avoid log(0)
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    return np.mean(-(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))


def binary_crossentropy_derivative(yhat: float, y: float) -> float:
    """
    Derivative of the Binary Cross Entropy Loss Function.

    Args:
        yhat (float): Prediction.
        y (float): True Label.
    """
    epsilon = 1e-15  # added to prevent division by 0 error
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    return (yhat - y) / (yhat * (1 - yhat))