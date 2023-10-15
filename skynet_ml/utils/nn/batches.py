from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class MiniBatchCreator(ABC):
    """
    Abstract base class for mini-batch creation. Provides a blueprint for creating mini-batches from datasets.

    Derived classes should implement the 'create' method to return mini-batches from the input data.
    """


    @abstractmethod
    def create(self, X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Create mini-batches from the given dataset.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector or matrix.
            batch_size (int): Size of each mini-batch.

        Returns:
            List[Tuple[np.array, np.array]]: List of mini-batches where each mini-batch is a tuple of (X_mini, y_mini).
        """
        pass


class DefaultMiniBatchCreator(MiniBatchCreator):
    """
    Default implementation of the MiniBatchCreator. Creates mini-batches by shuffling the input data and then 
    splitting it based on the specified batch size.

    This class uses a static method to ensure that no instance is needed to create mini-batches.
    """


    @staticmethod
    def create(X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Create mini-batches from the given dataset using the default shuffling and splitting method.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector or matrix.
            batch_size (int): Size of each mini-batch.

        Raises:
            AssertionError: If the number of samples in X and y do not match.

        Returns:
            List[Tuple[np.array, np.array]]: List of mini-batches where each mini-batch is a tuple of (X_mini, y_mini).
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
