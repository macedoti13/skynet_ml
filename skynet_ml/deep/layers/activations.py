import numpy as np

def linear(x: float) -> float:
    """
    Linear function.
    """
    return x


def d_linear(x: float) -> int:
    """
    Derivative of the linear function.
    """
    return 1


def sigmoid(x: float) -> float:
    """
    Sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    """
    Derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: float) -> float:
    """
    Relu activation function.
    """
    return np.maximum(0, x)


def d_relu(x: float) -> int:
    """
    Derivative of the relu function.
    """
    return np.where(x < 0, 0, 1)


def tanh(x: float) -> float:
    """
    Hyperbolic tangent activaiton function.
    """
    return np.tanh(x)


def d_tanh(x: float) -> float:
    """
    Derivative of the tanh function.
    """
    return 1 - np.tanh(x)**2


def softmax(s: np.array) -> np.array:
    """
    Softmax activation function.
    """
    exp_s = np.exp(s - np.max(s))  
    return exp_s / np.sum(exp_s)


def d_softmax(z: np.array) -> np.array:
    """
    Derivative of the softmax function.
    """
    n, batch_size = z.shape
    gradient = np.zeros_like(z)
    
    for i in range(batch_size):
        zi = z[:, i] 
        pi = softmax(zi)
        gradient[:, i] = pi * (1 - pi)  
    
    return gradient


activations_map = {
    "linear": linear,
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "softmax": softmax
}


d_activations_map = {
    "linear": d_linear,
    "sigmoid": d_sigmoid,
    "relu": d_relu,
    "tanh": d_tanh,
    "softmax": d_softmax
}
