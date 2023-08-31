import numpy as np
from typing import Callable, Any

def get_loss_function(loss: str) -> Callable[..., Any]:
    """
    Returns the loss function corresponding to the provided name.
    """
    if loss == "mse":
        return mse


def get_loss_derivative(loss: str) -> Callable[..., Any]:
    """
    Returns the derivative of the loss function corresponding to the provided name.
    """
    if loss == "mse":
        return mse_derivative
    
    
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