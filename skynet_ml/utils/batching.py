from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class MiniBatchCreator(ABC):
    """
    Abstract base class defining the interface for creating mini-batches from data.
    """

    @abstractmethod
    def create(self, X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Abstract method to create mini-batches from the provided data.

        Parameters:
        - X (np.array): Input data.
        - y (np.array): Ground truth labels.
        - batch_size (int): Desired size of each mini-batch.

        Returns:
        - List[Tuple[np.array, np.array]]: List of tuples, where each tuple contains a mini-batch of data 
                                           and corresponding labels.
        """
        pass
    
    
class DefaultMiniBatchCreator(MiniBatchCreator):
    """
    Default implementation for creating mini-batches from data.
    """

    @staticmethod
    def create(X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Create mini-batches from the provided data.

        Args:
        - X (np.array): Input data.
        - y (np.array): True labels.
        - batch_size (int): Desired size of each mini-batch.

        Returns:
        - mini_batches (List[Tuple[np.array, np.array]]): List of tuples, where each tuple contains a mini-batch 
                                                          of data and corresponding labels.
        """
        
        assert X.shape[0] == y.shape[0], "Mismatch between the number of samples in X and y."

        # Create an array of indices from 0 to the number of samples.
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)  # Shuffle the indices.
        X = X[indices]  # Shuffle X using the shuffled indices.
        y = y[indices]  # Shuffle y using the shuffled indices.

        mini_batches = []

        total_batches = X.shape[0] // batch_size
        for i in range(total_batches):
            X_mini = X[i * batch_size: (i + 1) * batch_size]
            y_mini = y[i * batch_size: (i + 1) * batch_size]
            mini_batches.append((X_mini, y_mini))

        # Handle the end case (last mini-batch < mini_batch_size)
        if X.shape[0] % batch_size != 0:
            X_mini = X[total_batches * batch_size:]
            y_mini = y[total_batches * batch_size:]
            mini_batches.append((X_mini, y_mini))

        return mini_batches
