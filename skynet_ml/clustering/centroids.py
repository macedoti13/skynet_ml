import numpy as np
from typing import Callable, Any

def get_centroid_initialization_method(name: str) -> Callable[..., Any]:
    """
    Returns the centroid initializaition function corresponding to the provided name.
    """    
    if name == "random":
        return randomly_initialize_centroids
    
    
def randomly_initialize_centroids(X: np.array, n_centroids: int) -> np.array:
    """
    Creates centroids by randomly choosing n points from the input data. 

    Args:
        X (np.array): Input data. 
        n_centroids (int): Number of centroids to create.

    Returns:
        np.array: Matrix with the centroids.
    """    
    random_indices = np.random.choice(X.shape[0], n_centroids, replace=False)
    return X[random_indices]