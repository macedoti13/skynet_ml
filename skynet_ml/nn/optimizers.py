import numpy as np

def get_optimizer(name: str):
    """
    Returns the optimization function corresponding to the provided name.
    """
    if name == "sgd":
        return sgd_update


def sgd_update(parameter: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Updates parameters using Stochastic Gradient Descent (SGD).

    Args:
        parameter (np.ndarray): Current values of the parameters that need to be updated.
        gradient (np.ndarray): Gradients computed during the backpropagation step.
        learning_rate (float): The step size to use during the SGD update.

    Returns:
        np.ndarray: Updated parameters after applying the SGD step.
    """    
    return parameter - learning_rate * gradient
