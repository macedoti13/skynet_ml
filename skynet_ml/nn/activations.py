import numpy as np
from typing import Callable, Any

def get_activation_function(name: str) -> Callable[..., Any]:
    """
    Returns the activation function corresponding to the provided name.
    """
    if name == "sigmoid":
        return sigmoid
    
    
def get_activation_derivative(name: str) -> Callable[..., Any]:
    """
    Returns the derivative of the activation function corresponding to the provided name.
    """
    if name == "sigmoid":
        return sigmoid_derivative


def sigmoid(x: float) -> float:
    """
    Sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """
    Derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))

