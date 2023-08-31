import numpy as np
from typing import Callable, Any

def get_distance(name: str)  -> Callable[..., Any]:
    """
    Returns the distance function corresponding to the provided name.
    """
    if name == "euclidean":
        return euclidean
    elif name == "manhattan":
        return manhattan


def euclidean(X: np.array, y: np.array) -> np.array:
    """
    Euclidian distance.
    """
    return np.linalg.norm(X - y, axis=1)


def manhattan(X: np.array, y: np.array) -> np.array:
    """
    Manhattan distance.
    """
    return np.sum(np.abs(X - y), axis=1)
