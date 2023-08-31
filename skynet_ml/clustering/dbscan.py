import numpy as np 
from skynet_ml.clustering.distances import get_distance

class DBSCAN:
    
    
    def __init__(self, max_distance: float, min_points: int, distance_metric: str = "euclidean") -> None:
        """
        Initialize the DBSCAN clustering algorithm with the given parameters.
        
        Args:
            max_distance (float): The maximum distance between two samples for one to be considered as in the neighborhood 
                                of the other. Used to determine the direct reachability of data points.
            min_points (int): The number of samples (or total weight) in a neighborhood for a point to be considered as 
                            a core point. This includes the point itself.
            distance_metric (str, optional): The metric used to calculate distance between points. It can be any distance 
                                            metric supported by the `get_distance` function. Defaults to "euclidean".
        
        Attributes:
            max_distance (float): The maximum distance threshold.
            min_points (int): Minimum number of neighboring points for a point to be considered as a core point.
            distance (callable): A function to compute distance between points based on the chosen `distance_metric`.
            clusters (list): List of clusters formed after calling the `fit` method. Each cluster is a numpy array 
                            of points.
            noise (list): List of noise points detected after calling the `fit` method.
        """
        self.max_distance = max_distance
        self.min_points = min_points
        self.distance = get_distance(distance_metric)
        self.clusters = []
        self.noise = []
        
        
    def points_within_range(self, X: np.array, point: np.array) -> np.array:
        """
        Retrieve points from dataset `X` that are within `self.max_distance` from the analyzed `point`.

        Args:
            X (np.array): The dataset.
            point (np.array): The analyzed point.

        Returns:
            np.array: An array of points (inner arrays) from `X` that are within `self.max_distance` from the analyzed `point`.
        """     
        # calculates the distance between all points in X from point
        distances_to_point = self.distance(X, point)
        
        # filter only the points that are within self.max_distance from point
        within_range = X[distances_to_point <= self.max_distance]
        
        return within_range
        
        
    def expand_cluster(self, X: np.array, point: np.array, point_neighbours: np.array) -> np.array:
        """
        Expands a cluster from a given point. Given a point, we want to create a cluster containing all the points that are reachable from it. To know if a point is reachable from other point,
        it has to be within self.max_distance from this other point, or it has to be within self.max_distance from a core point that is reachable from this other point. Therefore, what this 
        function does is the following: Starting from point x, we create a cluster containing only this point x. Then, we go through every single neighbour n (n is within self.max_distance from x) 
        and check if it's a core point (there are at least self.min_points within self.max_distance from n). If it is, then all points that are reachable from this neighbour n are also reachable 
        from the original point x, therefore, they should belong to our cluster. So we take all this neighbour point neighbours (neighbours of n) and mark them as neighbour from x (because they 
        are reachable from x). Now, we just updated the list of neighbours from the orinal point x. In the end, we add the neighbours to our cluster. We do that until there are no more neighbours
        to look, returning the final cluster. 

        Args:
            X (np.array): The dataset. 
            point (np.array): The point we will start the cluster expansion from. All points reachable from this point will be on our cluster. 
            point_neighbours (np.array): Array of points that are directly reachable (within self.max_distance) from the original point.

        Returns:
            cluster (np.array): A numpy array of numpy arrays. Each inner array is a point that is reachable from the original point.
        """        
        # creates the cluster by creating an empty numpy array of shape (0 (lines), n_features (columns))
        n_features = point.shape[0] # point is of shape (x,) where x is the number of features
        cluster = np.empty((0, n_features))
        
        # Appends the initial point to the cluster
        cluster = np.vstack([cluster, point])
        
        # iterates through all the neighbors from the point, checking if they are a core point (then we add their neighbors to our cluster) or not (then they are a border point)
        for i, neighbour in enumerate(point_neighbours):
            neighbour_neighbours = self.points_within_range(X, neighbour)  # gives me a numpy array of numpy arrays

            # checks if the point is a core point (if it is, we want to add its neighbors to our cluster)
            if len(neighbour_neighbours) >= self.min_points:
                # if it is:
                # look at each neighbors neighbor
                for new_neighbour in neighbour_neighbours:
                    # if it's not already an original point neighbor and it's not yet in the cluster
                    # the same as "if new_neighbour not in point_neighbours and new_neighbour not in cluster:"
                    if not np.any(np.all(point_neighbours == new_neighbour, axis=1)) and not np.any(np.all(cluster == new_neighbour, axis=1)):
                        # add point to numpy array of the neighbors of the original point
                        point_neighbours = np.vstack([point_neighbours, new_neighbour])

            # if its not: add the point to my cluster (add the point as a border point)
            if not np.any(np.all(cluster == neighbour, axis=1)):
                cluster = np.vstack([cluster, neighbour])

        return cluster
    
    
    def create_noise_list(self, potential_noise_list: list) -> None:
        """
        Adds to the self.noise list all points that didn't end up in a cluster after the clustering process.

        Args:
            potential_noise_list (list): List of points (numpy arrays) that were added to the potential noise list.
        """        
        for point in potential_noise_list:
            in_cluster = False
            
            # iterates through each cluster 
            for cluster in self.clusters:
                
                # check if the point is inside a cluster after the end of the clustering process
                if np.any(np.all(cluster == point, axis=1)):
                    in_cluster = True
                    break
            
            # point didn't end up in a cluster, then add it to the list of noise points
            if not in_cluster:
                self.noise.append(point)


    def fit(self, X: np.array) -> None:
        """
        Fit DBSCAN model to the dataset X. Starts by marking all points as not visited. Them, for each point in the dataset, mark it as visited and check if it's a core point. If it is, start the 
        process of creating a cluster from it. Once the cluster is created, mark all the points in the cluster as visited so they don't get visited again in future iteractions. In the end, create 
        the noise list by checking if the points ended up in a cluster or not. 

        Args:
            X (np.array): The dataset to be clustered. 
        """        
        # Initialize a list to keep track of visited points
        visited = np.zeros(X.shape[0], dtype=bool)
        
        # initialize a list for potential noise points
        potential_noise = []
        
        # Iterate through each point in the dataset
        for idx, point in enumerate(X):    
            # check if the point hasn't been visited
            if not visited[idx]:
                # mark the point as visited
                visited[idx] = True
                # get all the neighbours from this point
                neighbours = self.points_within_range(X, point)
                
                # check if it's a core point
                if len(neighbours) >= self.min_points:
                    
                    # it's a core point, then we create a cluster from it
                    new_cluster = self.expand_cluster(X, point, neighbours)
                    
                    # add the new cluster to the list of clusters
                    self.clusters.append(new_cluster)
    
                    # Mark all points in the new_cluster as visited
                    for cluster_point in new_cluster:
                        cluster_idx = np.where(np.all(X == cluster_point, axis=1))[0][0]
                        visited[cluster_idx] = True
                        
                else:
                    # it's not a core point, then add the point as noise
                    potential_noise.append(point)
                    
        # Check each point in potential_noise. If it hasn't ended up in any cluster, add it to the noise list.
        self.create_noise_list(potential_noise)
        
        
    def fit_predict(self, X: np.array) -> np.array:
        """
        Predicts which cluster each point in X belongs to. If the point doesn't belong to any cluster, label it as noise (-1).

        Args:
            X (np.array): The new data points.

        Returns:
            labels (np.array): An array of cluster labels.
        """
        
        labels = []
        
        # Iterate over each new point
        for point in X:
            assigned = False  # Track if the point was assigned to a cluster

            # Check distance to each existing cluster
            for cluster_idx, cluster in enumerate(self.clusters):
                distances = self.distance(cluster, point)  # Calculate distances to all points in the cluster
                if np.min(distances) <= self.max_distance:
                    labels.append(cluster_idx)  # Assign to the cluster if within max_distance
                    assigned = True
                    break  # Break out of cluster checking loop

            # If the point wasn't assigned to any cluster, label as noise (-1)
            if not assigned:
                labels.append(-1)
        
        return np.array(labels)
