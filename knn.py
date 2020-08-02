from distance import calDistance 
import numpy as np

def knn_naive(k, data, point):
    distances = np.zeros((len(data), len(data[0])+1))
    results = np.zeros((k, len(data[0])))
    for i in range(len(data)):
        dist = calDistance(point, data[i], "Cosine")
        for j in range(len(results[0])):
            distances[i][j] = data[i][j]
        distances[i][-1] = dist
    distances = distances[distances[:,-1].argsort()]
    for i in range(k):
         for j in range(len(results[0])):
            results[i][j] = distances[i][j]
    return results, distances[:k,-1]