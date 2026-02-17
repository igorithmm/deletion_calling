import numpy as np

def kmeans(X, k, maxIt):
    """
    K-means clustering algorithm
    
    Args:
        X: Input data dictionary/array (n_samples, n_features)
        k: Number of clusters
        maxIt: Maximum iterations
        
    Returns:
        Array with cluster labels in the last column
    """
    numPoints, numDim = X.shape
    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X
    
    # Initialize centroids randomly
    # Note: potential issue if k > numPoints, but matching original logic
    if numPoints >= k:
        centroids = dataSet[np.random.choice(numPoints, k, replace=False), :]
    else:
        # Fallback if fewer points than clusters (edge case)
        centroids = dataSet[np.random.choice(numPoints, k, replace=True), :]
        
    centroids[:, -1] = range(1, k + 1)
    
    iterations = 0
    oldCentroids = None
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        oldCentroids = np.copy(centroids)
        iterations += 1
        updateLabels(dataSet, centroids)
        centroids = getCentroids(dataSet, k)
        
    return dataSet
    
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    if oldCentroids is None:
        return False
    return np.array_equal(oldCentroids, centroids)  

def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)
    
    
def getLabelFromClosestCentroid(dataSetRow, centroids):
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1 , centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    return label
    
def getCentroids(dataSet, k):
    # -1 is label column
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        # Filter for points in cluster i
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        
        if len(oneCluster) > 0:
            result[i - 1, :-1] = np.mean(oneCluster, axis = 0)
        else:
            # Handle empty cluster if necessary (inherited logic implicitly handled this or crashed?)
            # Original code: result[i - 1, :-1] = np.mean(oneCluster, axis = 0) -> simple mean
            pass
            
        result[i - 1, -1] = i
        
    result = np.nan_to_num(result)
    return result
