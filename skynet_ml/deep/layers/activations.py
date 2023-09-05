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
    Softmax Activation function.
    """    
    exp_s = np.exp(s - np.max(s, axis=0))
    return exp_s / np.sum(exp_s, axis=0)


def d_softmax_dummy(z: np.array) -> np.array:
    """
    This is a "dummy" derivative function for the softmax activation. When combined with the Categorical Cross-Entropy 
    (CCE) loss, the need to compute the explicit derivative of the softmax function is eliminated. This is due to the 
    unique mathematical interaction between the softmax and CCE that simplifies their combined derivative to just 
    yhat - y, where yhat is the softmax output and y is the true label.

    In our backpropagation process, we multiply the gradient of the loss with respect to the output (d_output) by the 
    derivative of the activation function. However, in the case of softmax + CCE, this multiplication effectively 
    becomes a no-op. To maintain a consistent interface in our code without introducing special conditions for softmax, 
    this dummy derivative function is introduced, which just returns ones. This ensures that the element-wise 
    multiplication in the backward function becomes neutral when using softmax activation.

    Args:
        z (np.array): The pre-activation values. Not used in this dummy function but kept for interface consistency.

    Returns:
        np.array: A matrix of ones with the same shape as z.
    """
    return np.ones_like(z)


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
    "softmax": d_softmax_dummy
}
