import numpy as np
from skynet_ml.clustering.distances import get_distance
from skynet_ml.clustering.centroids import get_centroid_initialization_method

class KMeans:
    """
    KMeans clustering model.
    
    Attributes: 
        k (int): Number of clusters the model will create.
        distance (function): The function that calculates the distance between sample and centroid.
        centroid_initialization_method (function): The function that creates the first centroids.
        centroids (np.array): The centroids created by the model.
    """    
    
    def __init__(self, n_clusters: int, distance_metric: str = "euclidean", centroid_initialization_method: str = "random") -> None:
        """
        Initializes the KMeans clustering model.

        Args:
            n_clusters (int): Number of clusters the model will create.
            distance_metric (str, optional): Distance used by the model. Defaults to "euclidian".
            centroid_initialization_method (str, optional): Method for the model to initialize the clusters. Defaults to "random".
        """        
        self.k = n_clusters
        self.distance = get_distance(distance_metric)
        self.centroid_initialization_method = get_centroid_initialization_method(centroid_initialization_method)
        
        
    def initialize_centroids(self, X: np.array) -> None:
        """
        Applies the centroid initialization method and creates the first centroids.

        Args:
            X (np.array): Input data.
        """        
        self.centroids = self.centroid_initialization_method(X, self.k)
        
        
    def fit(self, X: np.array, max_iters: int = 100, tolerance: float = 1e-4) -> None:
        """
        Fits the model to the data using the KMeans clustering algorithm.

        Args:
            X (np.array): Input data.
            max_iters (int, optional): Maximum number of iterations for the algorithm. Defaults to 100.
            tolerance (float, optional): Convergence threshold. Algorithm stops if the change is below this threshold. Defaults to 1e-4.
        """        
        # initializes the centroids with the initialization method specified in the creation of the object
        self.initialize_centroids(X)
        
        for _ in range(max_iters):
            
            # saves centroids before update for comparison 
            old_centroids = self.centroids.copy()
            
            # compute the distance between each sample and each centroid
            distances = np.array([self.distance(X, centroid) for centroid in self.centroids])
            
            # assign each sample to the nearest centroid, forming the clusters
            labels = np.argmin(distances, axis=0)
            
            # recalculate centroids
            for i in range(self.k):
                self.centroids[i] = X[labels == i].mean(axis=0)
                
            # check for convergence (no sufficient change was made in the centroids)
            if np.all(np.abs(self.centroids - old_centroids) < tolerance):
                break
            
        # calculate the inertia after the algorithm stops
        self.calculate_inertia(X)
            
            
    def predict(self, X: np.array) -> None:
        """
        Assigns each data point in X to the nearest centroid.

        Args:
            X (np.array): Input dataset.

        Returns:
            _type_: Array of cluster labels.
        """        
        # compute the distance between each sample and each centroid
        distances = np.array([self.distance(X, centroid) for centroid in self.centroids])
        
        # assign each sample to the nearest centroid, forming the clusters
        labels = np.argmin(distances, axis=0)
        
        return labels
    
    
    def calculate_inertia(self, X: np.array) -> None:
        """
        Calculate the inertia of the clustering.
        
        Args:
            X (np.array): Input data.
        
        Returns:
            float: The inertia of the current clustering.
        """
        # Get the labels assigned to each data point
        labels = self.predict(X)

        # For each data point, find the squared distance to its assigned centroid
        sum_of_squares = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            sum_of_squares += np.sum(np.square(cluster_points - self.centroids[i]))

        self._interia = sum_of_squares