import numpy as np
from typing import Callable, Any

def get_activation_function(name: str) -> Callable[..., Any]:
    """
    Returns the activation function corresponding to the provided name.
    """
    if name == "sigmoid":
        return sigmoid
    elif name == "linear":
        return linear
    elif name == "relu":
        return relu
    elif name == "tanh":
        return tanh
    
    
def get_activation_derivative(name: str) -> Callable[..., Any]:
    """
    Returns the derivative of the activation function corresponding to the provided name.
    """
    if name == "sigmoid":
        return sigmoid_derivative
    elif name == "linear":
        return linear_derivative
    elif name == "relu":
        return relu_derivative
    elif name == "tanh":
        return tanh_derivative


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


def linear(x: float) -> float:
    """
    Linear function.
    """
    return x


def linear_derivative(x: float) -> int:
    """
    Derivative of the linear function.
    """
    return 1


def relu(x: float) -> float:
    """
    Relu activation function.
    """
    return np.maximum(0, x)


def relu_derivative(x: float) -> int:
    """
    Derivative of the relu function.
    """
    return np.where(x < 0, 0, 1)


def tanh(x: float) -> float:
    """
    Hyperbolic tangent activaiton function.
    """
    return np.tanh(x)


def tanh_derivative(x: float) -> float:
    """
    Derivative of the tanh function
    """
    return 1 - np.tanh(x)**2