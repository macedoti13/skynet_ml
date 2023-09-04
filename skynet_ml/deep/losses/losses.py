import numpy as np


def mse(yhat: float, y: float) -> float:
    """
    Mean Squared Error Function.

    Args:
        yhat (float): Prediction.
        y (float): True label.
    """    
    return np.mean(0.5 * (yhat - y)**2)


def d_mse(yhat: float, y: float) -> float:
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


def d_binary_cross_entropy(yhat: float, y: float) -> float:
    """
    Derivative of the Binary Cross Entropy Loss Function.

    Args:
        yhat (float): Prediction.
        y (float): True Label.
    """
    epsilon = 1e-15  # added to prevent division by 0 error
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    return (yhat - y) / (yhat * (1 - yhat))


def categorical_cross_entropy(yhat: float, y: float) -> float:
    """
    Categorical Cross-Entropy Loss Function. Used for multiclass Classification.

    Args:
        yhat (float): Prediction.
        y (float): True Label.
        
    Returns:
        float: Loss value. Folows the formula for the CCE Loss.
    """    
    epsilon = 1e-15 # prevents division by 0 
    yhat = np.clip(yhat , epsilon , 1 - epsilon)
    return -np.mean(np.sum(y * np.log(yhat), axis=1))


def d_categorical_cross_entropy(yhat: float, y: float) -> float:
    """
    Derivative of the Categorical Cross Entropy Loss Function when combined with softmax.

    Args:
        yhat (float): Prediction.
        y (float): True Label.
    """    
    return yhat - y